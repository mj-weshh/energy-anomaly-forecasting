"""Grid search DBSCAN hyperparameters with temporal validation splits.

Enhanced mode (default): scaled features, finer eps grid, train/val/test metrics.
Legacy mode (--legacy): original coarse grid on full dataset without scaling.

Run from repository root::

    python scripts/tune_dbscan.py
    python scripts/tune_dbscan.py --legacy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.features.build_features import (  # noqa: E402
    build_all_features,
    build_enhanced_anomaly_features,
)
from src.models.anomaly_preprocessing import AnomalyPreprocessor  # noqa: E402
from src.models.evaluate_models import (  # noqa: E402
    evaluate_anomaly_model,
    evaluate_on_splits,
)
from src.models.train_anomaly_models import prepare_feature_matrix  # noqa: E402
from src.models.tuning_utils import (  # noqa: E402
    align_labels,
    temporal_train_val_test_split,
)

LEGACY_EPS_VALUES = [0.1, 0.3, 0.5, 0.7]
LEGACY_MIN_SAMPLES_VALUES = [5, 10, 20]

ENHANCED_EPS_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
ENHANCED_MIN_SAMPLES_VALUES = [3, 5, 10, 15, 20]
METRIC_VALUES = ["euclidean", "manhattan"]


def _scaled_matrix(df, train_idx):
    feature_matrix = prepare_feature_matrix(df)
    hours = df.loc[feature_matrix.index, "hour"]
    consumption = df.loc[feature_matrix.index, "Electricity_Consumed"]
    preprocessor = AnomalyPreprocessor()
    train_mask = np.zeros(len(feature_matrix), dtype=bool)
    train_mask[train_idx] = True
    preprocessor.fit(feature_matrix, hours, consumption, train_mask)
    return feature_matrix, preprocessor.transform(feature_matrix, hours, consumption)


def run_legacy(csv_path: str) -> None:
    df = build_all_features(load_smart_meter_data(csv_path))
    clean_index = prepare_feature_matrix(df).index
    y_true = align_labels(df, clean_index)

    best_f1 = -1.0
    best = None

    print(f"Loaded: {csv_path}")
    print(f"Evaluation rows: {len(y_true)}\n")
    print("DBSCAN legacy grid (full dataset, unscaled):")
    print(f"{'eps':>5}  {'min_samples':>11}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'pred_anom':>9}")
    print("-" * 52)

    feature_matrix = prepare_feature_matrix(df)
    X = feature_matrix.to_numpy(dtype=float)

    for eps in LEGACY_EPS_VALUES:
        for min_samples in LEGACY_MIN_SAMPLES_VALUES:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            preds = (model.fit_predict(X) == -1).astype(int)
            metrics = evaluate_anomaly_model(y_true, preds)
            print(
                f"{eps:5.1f}  {min_samples:11d}  "
                f"{metrics['precision']:6.3f}  {metrics['recall']:6.3f}  "
                f"{metrics['f1']:6.3f}  {int(preds.sum()):9d}"
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best = (eps, min_samples, metrics, int(preds.sum()))

    assert best is not None
    eps, min_samples, metrics, pred_count = best
    cm = metrics["confusion_matrix"]
    print("\nLegacy best F1:")
    print(f"  eps={eps}  min_samples={min_samples}  F1={metrics['f1']:.3f}")
    print(f"  Predicted anomalies: {pred_count} / {len(y_true)}")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")


def run_enhanced(csv_path: str) -> None:
    df = build_enhanced_anomaly_features(load_smart_meter_data(csv_path))
    feature_matrix = prepare_feature_matrix(df)
    y_true = align_labels(df, feature_matrix.index)
    train_idx, val_idx, test_idx = temporal_train_val_test_split(len(feature_matrix))
    _, X = _scaled_matrix(df, train_idx)

    best_val_f1 = -1.0
    best_config: dict = {}

    print(f"Loaded: {csv_path}")
    print(f"Evaluation rows: {len(y_true)}")
    print(f"Train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")
    print("DBSCAN enhanced grid (scaled, validation F1):")
    print(
        f"{'eps':>5}  {'min_s':>5}  {'metric':>10}  "
        f"{'val_F1':>7}  {'test_F1':>7}  {'pred':>6}"
    )
    print("-" * 52)

    for eps in ENHANCED_EPS_VALUES:
        for min_samples in ENHANCED_MIN_SAMPLES_VALUES:
            for metric in METRIC_VALUES:
                model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                preds = (model.fit_predict(X) == -1).astype(int)
                val_metrics = evaluate_anomaly_model(y_true[val_idx], preds[val_idx])
                test_metrics = evaluate_anomaly_model(y_true[test_idx], preds[test_idx])
                print(
                    f"{eps:5.2f}  {min_samples:5d}  {metric:>10}  "
                    f"{val_metrics['f1']:7.3f}  {test_metrics['f1']:7.3f}  "
                    f"{int(preds.sum()):6d}"
                )
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_config = {
                        "eps": eps,
                        "min_samples": min_samples,
                        "metric": metric,
                        "preds": preds,
                    }

    assert best_config
    split_metrics = evaluate_on_splits(
        y_true, best_config["preds"], train_idx, val_idx, test_idx
    )

    print("\nBest validation config:")
    print(f"  eps={best_config['eps']}  min_samples={best_config['min_samples']}")
    print(f"  metric={best_config['metric']}")
    print()
    print(f"{'Split':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for split_name, metrics in split_metrics.items():
        print(
            f"{split_name:<8} {metrics['precision']:10.3f} "
            f"{metrics['recall']:10.3f} {metrics['f1']:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune DBSCAN.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run original coarse full-dataset grid without scaling.",
    )
    args = parser.parse_args()

    csv_path = str(find_dataset_csv(get_project_root()))
    if args.legacy:
        run_legacy(csv_path)
    else:
        run_enhanced(csv_path)


if __name__ == "__main__":
    main()

"""Grid search Isolation Forest with temporal train/val/test splits.

Run from repository root::

    python scripts/tune_isolation_forest.py
    python scripts/tune_isolation_forest.py --drop-weather
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data  # noqa: E402
from src.features.build_features import build_enhanced_anomaly_features  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model, evaluate_on_splits  # noqa: E402
from src.models.tuning_utils import (  # noqa: E402
    find_best_threshold,
    isolation_forest_scores,
    predict_from_scores,
    prepare_temporal_tuning_data,
)

CONTAMINATION_VALUES = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
N_ESTIMATORS_VALUES = [100, 200, 300]
MAX_FEATURES_VALUES = [1.0, 0.8, 0.6]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Isolation Forest (enhanced).")
    parser.add_argument(
        "--drop-weather",
        action="store_true",
        help="Drop Temperature, Humidity, Wind_Speed from feature matrix.",
    )
    args = parser.parse_args()

    csv_path = find_dataset_csv(get_project_root())
    df = build_enhanced_anomaly_features(load_smart_meter_data(csv_path))
    data = prepare_temporal_tuning_data(df, scale=True, drop_weather=args.drop_weather)

    best_val_f1 = -1.0
    best_config: dict = {}
    best_threshold: float | None = None
    best_mode = "threshold"

    print(f"Loaded: {csv_path}")
    print(f"Evaluation rows: {len(data.y_true)}")
    print(
        f"Train/val/test: {len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)}\n"
    )
    print("Grid search on validation F1 (enhanced features, scaled):\n")

    for contamination in CONTAMINATION_VALUES:
        for n_estimators in N_ESTIMATORS_VALUES:
            for max_features in MAX_FEATURES_VALUES:
                model = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    max_features=max_features,
                    random_state=42,
                )
                model.fit(data.X[data.train_idx])

                val_scores = isolation_forest_scores(model, data.X[data.val_idx])
                threshold, val_metrics = find_best_threshold(
                    val_scores, data.y_true[data.val_idx]
                )
                val_f1_threshold = val_metrics["f1"]

                raw_val = model.predict(data.X[data.val_idx])
                val_preds_contam = (raw_val == -1).astype(int)
                contam_metrics = evaluate_anomaly_model(
                    data.y_true[data.val_idx], val_preds_contam
                )
                val_f1_contam = contam_metrics["f1"]

                if val_f1_threshold >= val_f1_contam and val_f1_threshold > best_val_f1:
                    best_val_f1 = val_f1_threshold
                    best_threshold = threshold
                    best_mode = "threshold"
                    best_config = {
                        "contamination": contamination,
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "drop_weather": args.drop_weather,
                    }
                elif val_f1_contam > best_val_f1:
                    best_val_f1 = val_f1_contam
                    best_threshold = None
                    best_mode = "contamination"
                    best_config = {
                        "contamination": contamination,
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "drop_weather": args.drop_weather,
                    }

    assert best_config

    final_model = IsolationForest(
        contamination=best_config["contamination"],
        n_estimators=best_config["n_estimators"],
        max_features=best_config["max_features"],
        random_state=42,
    )
    final_model.fit(data.X[data.train_idx])

    if best_mode == "threshold" and best_threshold is not None:
        all_preds = predict_from_scores(
            isolation_forest_scores(final_model, data.X), best_threshold
        )
    else:
        all_preds = (final_model.predict(data.X) == -1).astype(int)

    split_metrics = evaluate_on_splits(
        data.y_true, all_preds, data.train_idx, data.val_idx, data.test_idx
    )

    print("Best validation config:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"  mode: {best_mode}")
    if best_threshold is not None:
        print(f"  score_threshold: {best_threshold:.6f}")
    print()
    print(f"{'Split':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for split_name, metrics in split_metrics.items():
        print(
            f"{split_name:<8} {metrics['precision']:10.3f} "
            f"{metrics['recall']:10.3f} {metrics['f1']:10.3f}"
        )

    cm = split_metrics["test"]["confusion_matrix"]
    print("\nTest confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")


if __name__ == "__main__":
    main()

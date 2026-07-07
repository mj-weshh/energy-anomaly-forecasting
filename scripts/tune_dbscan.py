"""Grid search for DBSCAN hyperparameters against the benchmark label.

Loads smart meter data, applies feature engineering, and evaluates DBSCAN
across a small ``eps`` x ``min_samples`` grid. Labels are excluded from
training and used only for post-hoc F1 comparison.

Run from repository root::

    python scripts/tune_dbscan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.features.build_features import build_all_features  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model  # noqa: E402
from src.models.train_anomaly_models import (  # noqa: E402
    prepare_feature_matrix,
    train_dbscan,
)

EPS_VALUES = [0.1, 0.3, 0.5, 0.7]
MIN_SAMPLES_VALUES = [5, 10, 20]


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = build_all_features(load_smart_meter_data(csv_path))

    clean_index = prepare_feature_matrix(df).index
    y_true = (
        df.loc[clean_index, "Anomaly_Label"]
        .map({"Normal": 0, "Abnormal": 1})
        .to_numpy(dtype=int)
    )

    best_f1 = -1.0
    best_eps: float | None = None
    best_min_samples: int | None = None
    best_metrics = None
    best_pred_count = 0

    print(f"Loaded: {csv_path}")
    print(f"Evaluation rows: {len(y_true)}\n")
    print("DBSCAN grid search (Abnormal = positive class):")
    print(f"{'eps':>5}  {'min_samples':>11}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'pred_anom':>9}")
    print("-" * 52)

    for eps in EPS_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            _, y_pred = train_dbscan(df, eps=eps, min_samples=min_samples)
            metrics = evaluate_anomaly_model(y_true, y_pred)
            pred_count = int(y_pred.sum())

            print(
                f"{eps:5.1f}  {min_samples:11d}  "
                f"{metrics['precision']:6.3f}  {metrics['recall']:6.3f}  "
                f"{metrics['f1']:6.3f}  {pred_count:9d}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_eps = eps
                best_min_samples = min_samples
                best_metrics = metrics
                best_pred_count = pred_count

    assert best_metrics is not None and best_eps is not None and best_min_samples is not None
    cm = best_metrics["confusion_matrix"]

    print()
    print("DBSCAN grid search — best F1 (Abnormal = positive):")
    print(f"  eps={best_eps}  min_samples={best_min_samples}")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall:    {best_metrics['recall']:.3f}")
    print(f"  F1:        {best_metrics['f1']:.3f}")
    print(f"  Predicted anomalies: {best_pred_count} / {len(y_true)}")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")


if __name__ == "__main__":
    main()

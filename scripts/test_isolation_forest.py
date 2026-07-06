"""End-to-end Isolation Forest baseline verification.

Loads smart meter data, applies feature engineering, trains an unsupervised
Isolation Forest (labels excluded from training), and evaluates predictions
against the benchmark ``Anomaly_Label`` using imbalance-aware metrics.

Run from repository root::

    python scripts/test_isolation_forest.py
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
    train_isolation_forest,
)


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = build_all_features(load_smart_meter_data(csv_path))

    clean_index = prepare_feature_matrix(df).index
    y_true = (
        df.loc[clean_index, "Anomaly_Label"]
        .map({"Normal": 0, "Abnormal": 1})
        .to_numpy(dtype=int)
    )

    _, y_pred = train_isolation_forest(df)
    assert len(y_true) == len(y_pred), "y_true and y_pred length mismatch"

    metrics = evaluate_anomaly_model(y_true, y_pred)
    cm = metrics["confusion_matrix"]

    print(f"Loaded: {csv_path}")
    print(f"Evaluation rows: {len(y_true)}\n")
    print("Isolation Forest baseline metrics (Abnormal = positive class):")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")


if __name__ == "__main__":
    main()

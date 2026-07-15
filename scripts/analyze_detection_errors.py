"""Legacy IF false-positive analysis by hour on the temporal test split.

Run from repository root::

    python scripts/analyze_detection_errors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.features.build_features import build_all_features  # noqa: E402
from src.models.error_analysis import summarize_false_positives  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model  # noqa: E402
from src.models.feature_matrix import prepare_feature_matrix  # noqa: E402
from src.models.tuning_utils import align_labels, temporal_train_val_test_split  # noqa: E402

OUTPUT_PNG = REPO_ROOT / "docs" / "assets" / "anomaly" / "legacy_if_fp_by_hour.png"


def _legacy_test_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce legacy IF (test, production params) from compare_anomaly_models."""
    raw = load_smart_meter_data(find_dataset_csv(get_project_root()))
    df_legacy = build_all_features(raw)
    feature_matrix = prepare_feature_matrix(df_legacy)
    y_true = align_labels(df_legacy, feature_matrix.index)
    hours = df_legacy.loc[feature_matrix.index, "hour"].to_numpy(dtype=int)
    X = feature_matrix.to_numpy(dtype=float)
    train_idx, _, test_idx = temporal_train_val_test_split(len(feature_matrix))

    model = IsolationForest(
        contamination=0.05,
        n_estimators=100,
        max_features=1.0,
        random_state=42,
    )
    model.fit(X[train_idx])
    y_pred = (model.predict(X) == -1).astype(int)
    return hours, y_true, y_pred, test_idx


def main() -> None:
    hours, y_true, y_pred, test_idx = _legacy_test_predictions()
    metrics = evaluate_anomaly_model(y_true[test_idx], y_pred[test_idx])
    fp_table = summarize_false_positives(hours, y_true, y_pred, test_idx)

    print("Legacy IF (test, production params) — hourly false positives\n")
    print(f"Test F1: {metrics['f1']:.3f}  FP total: {metrics['confusion_matrix'][0, 1]}\n")
    print(fp_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    top_hours = fp_table.nlargest(5, "fp")
    if not top_hours.empty:
        print("\nTop hours by FP count:")
        for _, row in top_hours.iterrows():
            print(
                f"  hour {int(row['hour']):02d}: "
                f"FP={int(row['fp'])}, rate={row['fp_rate']:.3f}"
            )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fp_table["hour"], fp_table["fp"], color="#4C72B0")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("False positives (test split)")
    ax.set_title("Legacy IF — hourly false positives on temporal test window")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=120)
    plt.close(fig)
    print(f"\nSaved chart: {OUTPUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

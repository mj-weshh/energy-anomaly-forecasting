"""Per-segment enhanced IF evaluation on the temporal test split.

Trains one global enhanced IF (BEST_IF_CONFIG) and reports test F1 by hour
and weekend flag — no per-segment re-tuning (avoids overfitting sparse slices).

Run from repository root::

    python scripts/tune_isolation_forest_by_segment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data  # noqa: E402
from src.features.build_features import build_enhanced_anomaly_features  # noqa: E402
from src.models.anomaly_config import BEST_IF_CONFIG, TUNING_METRICS  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model  # noqa: E402
from src.models.tuning_utils import (  # noqa: E402
    isolation_forest_scores,
    predict_from_scores,
    prepare_temporal_tuning_data,
)

MIN_SEGMENT_ROWS = 30
GLOBAL_RECALL = TUNING_METRICS["enhanced_if_test_recall"]


def _segment_table(
    segment_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    *,
    label_map: dict[float | int, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    test_segments = segment_values[test_idx]

    for segment in sorted(np.unique(test_segments)):
        mask = test_segments == segment
        if mask.sum() < MIN_SEGMENT_ROWS:
            continue
        yt = y_true[test_idx][mask]
        yp = y_pred[test_idx][mask]
        metrics = evaluate_anomaly_model(yt, yp)
        cm = metrics["confusion_matrix"]
        label = label_map.get(segment, str(segment)) if label_map else str(segment)
        rows.append(
            {
                "segment": label,
                "n": int(mask.sum()),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "flag_low_recall": metrics["recall"] < GLOBAL_RECALL,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_smart_meter_data(find_dataset_csv(get_project_root()))
    df = build_enhanced_anomaly_features(raw)
    data = prepare_temporal_tuning_data(df, scale=True)

    model = IsolationForest(
        contamination=BEST_IF_CONFIG["contamination"],
        n_estimators=BEST_IF_CONFIG["n_estimators"],
        max_features=BEST_IF_CONFIG["max_features"],
        random_state=42,
    )
    model.fit(data.X[data.train_idx])
    preds = predict_from_scores(
        isolation_forest_scores(model, data.X),
        BEST_IF_CONFIG["score_threshold"],
    )

    global_test = evaluate_anomaly_model(
        data.y_true[data.test_idx], preds[data.test_idx]
    )
    print(f"Global enhanced IF test F1: {global_test['f1']:.3f}  recall: {global_test['recall']:.3f}\n")

    eval_index = data.feature_matrix.index
    hours = df.loc[eval_index, "hour"].to_numpy(dtype=int)
    weekends = df.loc[eval_index, "is_weekend"].to_numpy(dtype=int)

    print("Per-hour test breakdown (n >= 30):")
    hour_table = _segment_table(hours, data.y_true, preds, data.test_idx)
    print(hour_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nWeekend vs weekday test breakdown:")
    weekend_table = _segment_table(
        weekends,
        data.y_true,
        preds,
        data.test_idx,
        label_map={0: "weekday", 1: "weekend"},
    )
    print(weekend_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    low_recall = hour_table[hour_table["flag_low_recall"]] if not hour_table.empty else hour_table
    if not low_recall.empty:
        print("\nHours with recall below global enhanced test (0.364):")
        for _, row in low_recall.iterrows():
            print(f"  hour {row['segment']}: recall={row['recall']:.3f}, FP={int(row['fp'])}")


if __name__ == "__main__":
    main()

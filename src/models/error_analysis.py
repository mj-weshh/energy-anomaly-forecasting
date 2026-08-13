"""Error analysis helpers for anomaly detection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_by_hour(
    hours: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Count TP, FP, FN, TN per hour on the temporal test slice.

    Args:
        hours: Hour-of-day values for the full evaluation frame.
        y_true: Ground-truth labels (0/1) for all rows.
        y_pred: Model predictions (0/1) for all rows.
        test_idx: Indices selecting the temporal test slice.

    Returns:
        DataFrame with columns ``hour``, ``tn``, ``fp``, ``fn``, ``tp``,
        and ``n``, sorted by hour.
    """
    hour_values = np.asarray(hours, dtype=int)[test_idx]
    yt = y_true[test_idx]
    yp = y_pred[test_idx]

    records: list[dict[str, int]] = []
    for hour in range(24):
        mask = hour_values == hour
        if not mask.any():
            continue
        yt_h = yt[mask]
        yp_h = yp[mask]
        records.append(
            {
                "hour": hour,
                "tn": int(np.sum((yt_h == 0) & (yp_h == 0))),
                "fp": int(np.sum((yt_h == 0) & (yp_h == 1))),
                "fn": int(np.sum((yt_h == 1) & (yp_h == 0))),
                "tp": int(np.sum((yt_h == 1) & (yp_h == 1))),
                "n": int(mask.sum()),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("hour").reset_index(drop=True)


def summarize_false_positives(
    hours: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Hourly false-positive counts and rates vs normal rows in that hour.

    Args:
        hours: Hour-of-day values for the full evaluation frame.
        y_true: Ground-truth labels (0/1) for all rows.
        y_pred: Model predictions (0/1) for all rows.
        test_idx: Indices selecting the temporal test slice.

    Returns:
        DataFrame with ``hour``, ``fp``, ``normal_rows``, ``fp_rate``,
        ``fn``, ``tp``, and ``n``. Empty if the hourly confusion table is
        empty.
    """
    hourly = confusion_by_hour(hours, y_true, y_pred, test_idx)
    if hourly.empty:
        return hourly

    hourly["normal_rows"] = hourly["tn"] + hourly["fp"]
    hourly["fp_rate"] = hourly["fp"] / hourly["normal_rows"].replace(0, np.nan)
    return hourly[["hour", "fp", "normal_rows", "fp_rate", "fn", "tp", "n"]]


def fair_comparison_table() -> pd.DataFrame:
    """Return documented fair-comparison F1 scores from anomaly_config.

    Returns:
        Two-column DataFrame (``model``, ``test_f1``) of documented fair-
        comparison scores from ``TUNING_METRICS``.
    """
    from src.models.anomaly_config import TUNING_METRICS

    rows = [
        ("Legacy IF (full dataset)", TUNING_METRICS["legacy_if_full_dataset_f1"]),
        ("Legacy IF (test, production)", TUNING_METRICS["legacy_if_test_f1"]),
        (
            "Legacy IF (test, val threshold)",
            TUNING_METRICS["legacy_if_test_val_threshold_f1"],
        ),
        ("Enhanced IF (test)", TUNING_METRICS["enhanced_if_test_f1"]),
    ]
    return pd.DataFrame(rows, columns=["model", "test_f1"])

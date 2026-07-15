"""Tests for anomaly detection error analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.error_analysis import confusion_by_hour, summarize_false_positives


def test_confusion_by_hour_aggregates_fp() -> None:
    hours = np.array([0, 0, 1, 1, 1])
    y_true = np.array([0, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])
    test_idx = np.arange(5)

    result = confusion_by_hour(hours, y_true, y_pred, test_idx)

    hour0 = result.loc[result["hour"] == 0].iloc[0]
    assert hour0["fp"] == 1
    assert hour0["tp"] == 1
    assert hour0["fn"] == 0

    hour1 = result.loc[result["hour"] == 1].iloc[0]
    assert hour1["fp"] == 1
    assert hour1["fn"] == 1
    assert hour1["tp"] == 0


def test_summarize_false_positives_includes_rate() -> None:
    hours = np.array([2, 2, 2, 2])
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0])
    test_idx = np.arange(4)

    summary = summarize_false_positives(hours, y_true, y_pred, test_idx)
    row = summary.loc[summary["hour"] == 2].iloc[0]

    assert row["fp"] == 2
    assert row["normal_rows"] == 3
    assert abs(row["fp_rate"] - (2 / 3)) < 1e-9

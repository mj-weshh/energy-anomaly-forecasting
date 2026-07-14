"""Imbalance-aware evaluation for Phase 2 anomaly detection.

``Anomaly_Label`` is a benchmark only — models train unsupervised and
evaluation uses precision, recall, and F1 on the Abnormal class rather
than accuracy. See docs/phase2-strategy.md for rationale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_anomaly_model(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, Any]:
    """Compute imbalance-aware metrics for anomaly detection.

    Treats class ``1`` as the positive (Abnormal) class, matching the
    benchmark label encoding used in Phase 2 evaluation.

    Args:
        y_true: Ground-truth labels (0 = Normal, 1 = Abnormal).
        y_pred: Model predictions (0 = Normal, 1 = Abnormal).

    Returns:
        Dictionary with keys:
        ``precision``, ``recall``, ``f1`` (floats, Abnormal as positive),
        and ``confusion_matrix`` (2x2 ``numpy.ndarray`` in sklearn order:
        [[TN, FP], [FN, TP]]).

    Raises:
        ValueError: If inputs are empty or contain values other than 0/1.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if not set(np.unique(y_true_arr)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1.")
    if not set(np.unique(y_pred_arr)).issubset({0, 1}):
        raise ValueError("y_pred must contain only 0 and 1.")

    return {
        "precision": precision_score(
            y_true_arr, y_pred_arr, pos_label=1, zero_division=0
        ),
        "recall": recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]),
    }


def evaluate_on_splits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return train/val/test metric dicts for temporal evaluation."""
    return {
        "train": evaluate_anomaly_model(y_true[train_idx], y_pred[train_idx]),
        "val": evaluate_anomaly_model(y_true[val_idx], y_pred[val_idx]),
        "test": evaluate_anomaly_model(y_true[test_idx], y_pred[test_idx]),
    }

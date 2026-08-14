"""Temporal split and threshold utilities for anomaly model tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.anomaly_config import TEMPORAL_SPLIT_RATIOS
from src.models.anomaly_preprocessing import AnomalyPreprocessor
from src.models.evaluate_models import evaluate_anomaly_model
from src.models.feature_matrix import apply_feature_ablation, prepare_feature_matrix


@dataclass(frozen=True)
class TemporalTuningData:
    """Prepared arrays and indices for temporal anomaly tuning.

    Attributes:
        feature_matrix: Numeric feature frame after ablation / NaN drop.
        X: Model matrix (scaled when a preprocessor is used).
        y_true: Benchmark labels (0 = Normal, 1 = Abnormal).
        train_idx: Chronological training row indices.
        val_idx: Chronological validation row indices.
        test_idx: Chronological test row indices.
        preprocessor: Fitted ``AnomalyPreprocessor`` when ``scale=True``, else
            ``None``.
    """

    feature_matrix: pd.DataFrame
    X: np.ndarray
    y_true: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    preprocessor: AnomalyPreprocessor | None


LABEL_MAP = {"Normal": 0, "Abnormal": 1}


def align_labels(df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Map benchmark ``Anomaly_Label`` values to 0/1 for aligned rows.

    Args:
        df: DataFrame containing the ``Anomaly_Label`` column.
        index: Row index subset to align (typically the feature matrix index).

    Returns:
        1-D int array of mapped labels (0 = Normal, 1 = Abnormal).

    Raises:
        KeyError: If ``Anomaly_Label`` is missing.
        ValueError: If any label value is not in ``LABEL_MAP``.
    """
    if "Anomaly_Label" not in df.columns:
        raise KeyError("Anomaly_Label column not found in DataFrame.")

    labels = df.loc[index, "Anomaly_Label"]
    mapped = labels.map(LABEL_MAP)
    if mapped.isna().any():
        unknown = sorted(labels[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unmapped Anomaly_Label values: {unknown}")

    return mapped.to_numpy(dtype=int)


def temporal_train_val_test_split(
    n: int,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological index arrays for train, validation, and test.

    Args:
        n: Total number of rows to split.
        train_ratio: Fraction for training. Defaults to
            ``TEMPORAL_SPLIT_RATIOS['train']``.
        val_ratio: Fraction for validation. Defaults to
            ``TEMPORAL_SPLIT_RATIOS['val']``. The remainder is test.

    Returns:
        Tuple of ``(train_idx, val_idx, test_idx)`` as 1-D int arrays.

    Raises:
        ValueError: If ``n`` is not positive or
            ``train_ratio + val_ratio >= 1``.
    """
    if train_ratio is None:
        train_ratio = float(TEMPORAL_SPLIT_RATIOS["train"])
    if val_ratio is None:
        val_ratio = float(TEMPORAL_SPLIT_RATIOS["val"])

    if n <= 0:
        raise ValueError("n must be positive.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    all_idx = np.arange(n)
    return all_idx[:train_end], all_idx[train_end:val_end], all_idx[val_end:]


def prepare_temporal_tuning_data(
    df: pd.DataFrame,
    *,
    scale: bool = True,
    drop_weather: bool = False,
) -> TemporalTuningData:
    """Build scaled matrix, labels, and chronological splits for tuning scripts.

    Args:
        df: Feature-engineered DataFrame with labels and context columns.
        scale: If True, fit ``AnomalyPreprocessor`` on the train slice only.
        drop_weather: If True, drop weather columns before modeling.

    Returns:
        ``TemporalTuningData`` bundle ready for threshold / model sweeps.
    """
    feature_matrix = apply_feature_ablation(prepare_feature_matrix(df), drop_weather)
    y_true = align_labels(df, feature_matrix.index)
    train_idx, val_idx, test_idx = temporal_train_val_test_split(len(feature_matrix))

    preprocessor: AnomalyPreprocessor | None = None
    if scale:
        hours = df.loc[feature_matrix.index, "hour"]
        consumption = df.loc[feature_matrix.index, "Electricity_Consumed"]
        preprocessor = AnomalyPreprocessor()
        train_mask = np.zeros(len(feature_matrix), dtype=bool)
        train_mask[train_idx] = True
        preprocessor.fit(feature_matrix, hours, consumption, train_mask)
        X = preprocessor.transform(feature_matrix, hours, consumption)
    else:
        X = feature_matrix.to_numpy(dtype=float)

    return TemporalTuningData(
        feature_matrix=feature_matrix,
        X=X,
        y_true=y_true,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        preprocessor=preprocessor,
    )


def isolation_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores where higher values indicate more anomalous.

    Args:
        model: Fitted ``IsolationForest``.
        X: Feature matrix scored by ``model.decision_function``.

    Returns:
        1-D float array of inverted decision scores (higher = more anomalous).
    """
    return -model.decision_function(X)


def predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from score threshold.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        threshold: Cutoff; scores at or above the threshold become Abnormal.

    Returns:
        1-D int array (0 = Normal, 1 = Abnormal).
    """
    return (scores >= threshold).astype(int)


def find_best_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[float, dict[str, Any]]:
    """Sweep score quantiles on validation labels; return best-F1 threshold.

    Args:
        scores: Validation anomaly scores aligned with ``y_true``.
        y_true: Ground-truth labels (0/1) for the validation slice.
        n_thresholds: Maximum candidate thresholds when sampling quantiles.

    Returns:
        Tuple of ``(best_threshold, best_metrics)`` where ``best_metrics`` is
        the ``evaluate_anomaly_model`` dict for that threshold.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if scores.size == 0:
        raise ValueError("scores must be non-empty.")

    unique_scores = np.unique(scores)
    if unique_scores.size > n_thresholds:
        quantiles = np.linspace(0.01, 0.99, n_thresholds)
        candidates = np.unique(np.quantile(scores, quantiles))
    else:
        candidates = unique_scores

    best_f1 = -1.0
    best_threshold = float(candidates[0])
    best_metrics = evaluate_anomaly_model(y_true, predict_from_scores(scores, best_threshold))

    for threshold in candidates:
        preds = predict_from_scores(scores, float(threshold))
        metrics = evaluate_anomaly_model(y_true, preds)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1].

    Args:
        scores: Raw anomaly scores.

    Returns:
        Scores scaled to ``[0, 1]``. Returns zeros if all scores are equal.
    """
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)

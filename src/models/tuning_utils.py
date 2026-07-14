"""Temporal split and threshold utilities for anomaly model tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.evaluate_models import evaluate_anomaly_model

WEATHER_COLUMNS = ("Temperature", "Humidity", "Wind_Speed")


def align_labels(df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Map benchmark ``Anomaly_Label`` values to 0/1 for aligned rows."""
    return (
        df.loc[index, "Anomaly_Label"]
        .map({"Normal": 0, "Abnormal": 1})
        .to_numpy(dtype=int)
    )


def temporal_train_val_test_split(
    n: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological index arrays for train, validation, and test."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    all_idx = np.arange(n)
    return all_idx[:train_end], all_idx[train_end:val_end], all_idx[val_end:]


def apply_feature_ablation(
    feature_matrix: pd.DataFrame,
    drop_weather: bool = True,
) -> pd.DataFrame:
    """Drop weather columns for ablation experiments."""
    if not drop_weather:
        return feature_matrix
    drop_cols = [c for c in WEATHER_COLUMNS if c in feature_matrix.columns]
    return feature_matrix.drop(columns=drop_cols)


def isolation_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores where higher values indicate more anomalous."""
    return -model.decision_function(X)


def predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from score threshold."""
    return (scores >= threshold).astype(int)


def find_best_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[float, dict[str, Any]]:
    """Sweep score quantiles on validation labels; return best-F1 threshold."""
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
    """Min-max normalize scores to [0, 1]."""
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)

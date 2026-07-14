"""Unsupervised anomaly detection model training for Phase 2.

Isolation Forest and DBSCAN train on engineered features only.
``Anomaly_Label`` is excluded before fitting and is used strictly as an
evaluation benchmark — see docs/phase2-strategy.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from src.features.build_features import build_all_features
from src.models.feature_matrix import prepare_feature_matrix, prepare_model_matrix
from src.models.tuning_utils import (
    isolation_forest_scores,
    normalize_scores,
    predict_from_scores,
)

# Backward-compatible re-export for existing imports.
__all__ = [
    "detect_anomalies",
    "prepare_feature_matrix",
    "train_dbscan",
    "train_ensemble",
    "train_isolation_forest",
]


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 100,
    max_features: float | int = 1.0,
    scale: bool = False,
    score_threshold: float | None = None,
    preprocessor: Any = None,
    fit_indices: np.ndarray | None = None,
    drop_weather: bool = False,
) -> tuple[IsolationForest, np.ndarray]:
    """Train an Isolation Forest on engineered smart meter features.

    ``Anomaly_Label`` is never passed to the model. Predictions use the
    evaluation encoding: ``0`` = Normal, ``1`` = Abnormal.

    When ``fit_indices`` is set, the model fits on that chronological subset
    only (for temporal tuning). Legacy callers omit it and train on all rows.
    """
    _, X, _ = prepare_model_matrix(
        df,
        scale=scale,
        drop_weather=drop_weather,
        fit_indices=fit_indices,
        preprocessor=preprocessor,
    )

    X_fit = X if fit_indices is None else X[fit_indices]

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
    )
    model.fit(X_fit)

    if score_threshold is not None:
        scores = isolation_forest_scores(model, X)
        predictions = predict_from_scores(scores, score_threshold)
    else:
        raw_predictions = model.predict(X)
        predictions = (raw_predictions == -1).astype(int)

    return model, predictions


def train_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    scale: bool = False,
    preprocessor: Any = None,
    fit_indices: np.ndarray | None = None,
    drop_weather: bool = False,
) -> tuple[DBSCAN, np.ndarray]:
    """Train DBSCAN on engineered smart meter features."""
    _, X, _ = prepare_model_matrix(
        df,
        scale=scale,
        drop_weather=drop_weather,
        fit_indices=fit_indices,
        preprocessor=preprocessor,
    )

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    raw_labels = model.fit_predict(X)
    predictions = (raw_labels == -1).astype(int)
    return model, predictions


def train_ensemble(
    df: pd.DataFrame,
    if_kwargs: dict[str, Any] | None = None,
    dbscan_kwargs: dict[str, Any] | None = None,
    strategy: str = "intersection",
    alpha: float = 0.7,
    score_threshold: float | None = None,
    fit_indices: np.ndarray | None = None,
    drop_weather: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    """Combine Isolation Forest and DBSCAN predictions."""
    if_kwargs = dict(if_kwargs or {})
    dbscan_kwargs = dict(dbscan_kwargs or {})

    shared = {
        "fit_indices": fit_indices,
        "drop_weather": drop_weather,
    }
    if_kwargs.setdefault("scale", True)
    dbscan_kwargs.setdefault("scale", True)

    if_model, if_preds = train_isolation_forest(
        df, score_threshold=score_threshold, **shared, **if_kwargs
    )
    db_model, db_preds = train_dbscan(df, **shared, **dbscan_kwargs)

    strategy_key = strategy.strip().lower()
    if strategy_key == "union":
        ensemble_preds = np.maximum(if_preds, db_preds)
    elif strategy_key == "intersection":
        ensemble_preds = np.minimum(if_preds, db_preds)
    elif strategy_key == "weighted":
        _, X, _ = prepare_model_matrix(
            df,
            scale=if_kwargs.get("scale", True),
            drop_weather=drop_weather,
            fit_indices=fit_indices,
            preprocessor=if_kwargs.get("preprocessor"),
        )
        if_scores = normalize_scores(isolation_forest_scores(if_model, X))
        combined = alpha * if_scores + (1.0 - alpha) * db_preds.astype(float)
        ensemble_preds = (combined >= 0.5).astype(int)
    else:
        raise ValueError(
            f"Unsupported ensemble strategy {strategy!r}. "
            "Use 'intersection', 'union', or 'weighted'."
        )

    bundle = {"isolation_forest": if_model, "dbscan": db_model, "strategy": strategy_key}
    return bundle, ensemble_preds


def detect_anomalies(
    df: pd.DataFrame,
    model_type: str = "isolation_forest",
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    **kwargs: object,
) -> tuple[Any, np.ndarray]:
    """Route feature-engineered data to an unsupervised anomaly detector."""
    if feature_builder is not None:
        df = feature_builder(df)
    elif "hour" not in df.columns:
        df = build_all_features(df)

    model_key = model_type.strip().lower()
    if model_key == "isolation_forest":
        return train_isolation_forest(df, **kwargs)
    if model_key == "dbscan":
        return train_dbscan(df, **kwargs)
    if model_key == "ensemble":
        return train_ensemble(df, **kwargs)
    raise ValueError(
        f"Unsupported model_type {model_type!r}. "
        "Use 'isolation_forest', 'dbscan', or 'ensemble'."
    )

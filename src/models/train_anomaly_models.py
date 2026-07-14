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
from src.models.anomaly_preprocessing import AnomalyPreprocessor
from src.models.tuning_utils import (
    apply_feature_ablation,
    isolation_forest_scores,
    normalize_scores,
    predict_from_scores,
)

EXCLUDE_COLUMNS = frozenset({"Timestamp", "Anomaly_Label"})


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric feature matrix for unsupervised training.

    Drops ``Timestamp`` and ``Anomaly_Label``, then removes rows with NaN
    (rolling warm-up rows from ``add_rolling_metrics``).

    Args:
        df: Feature-engineered DataFrame from ``src.features.build_features``.

    Returns:
        Numeric feature matrix ready for unsupervised model fitting.

    Raises:
        ValueError: If no rows remain after dropping NaN values.
    """
    features = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])
    features = features.dropna()
    if features.empty:
        raise ValueError("No rows remain after dropping NaN feature values.")
    return features


def _aligned_context(
    df: pd.DataFrame, feature_matrix: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    index = feature_matrix.index
    hours = df.loc[index, "hour"]
    consumption = df.loc[index, "Electricity_Consumed"]
    return hours, consumption


def _prepare_matrix(
    df: pd.DataFrame,
    *,
    scale: bool,
    drop_weather: bool,
    fit_indices: np.ndarray | None,
    preprocessor: AnomalyPreprocessor | None,
) -> tuple[pd.DataFrame, np.ndarray, AnomalyPreprocessor | None]:
    feature_matrix = apply_feature_ablation(prepare_feature_matrix(df.copy()), drop_weather)
    hours, consumption = _aligned_context(df, feature_matrix)

    if scale:
        if fit_indices is None:
            fit_indices = np.arange(len(feature_matrix))
        if preprocessor is None:
            preprocessor = AnomalyPreprocessor()
        train_mask = np.zeros(len(feature_matrix), dtype=bool)
        train_mask[fit_indices] = True
        preprocessor.fit(feature_matrix, hours, consumption, train_mask)
        X = preprocessor.transform(feature_matrix, hours, consumption)
    else:
        X = feature_matrix.to_numpy(dtype=float)

    return feature_matrix, X, preprocessor


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 100,
    max_features: float | int = 1.0,
    scale: bool = False,
    score_threshold: float | None = None,
    preprocessor: AnomalyPreprocessor | None = None,
    fit_indices: np.ndarray | None = None,
    drop_weather: bool = False,
) -> tuple[IsolationForest, np.ndarray]:
    """Train an Isolation Forest on engineered smart meter features.

    ``Anomaly_Label`` is never passed to the model. Predictions use the
    evaluation encoding: ``0`` = Normal, ``1`` = Abnormal.

    When ``fit_indices`` is set, the model fits on that chronological subset
    only (for temporal tuning). Legacy callers omit it and train on all rows.

    Args:
        df: Feature-engineered DataFrame.
        contamination: Expected outlier proportion (used when ``score_threshold``
            is ``None``).
        random_state: Random seed.
        n_estimators: Number of isolation trees.
        max_features: Features per tree split.
        scale: If ``True``, apply train-fitted ``AnomalyPreprocessor``.
        score_threshold: If set, classify via ``decision_function`` threshold.
        preprocessor: Optional pre-fitted preprocessor (tuning reuse).
        fit_indices: Row indices for unsupervised fit and scaler fit.
        drop_weather: Drop weather columns before training.

    Returns:
        Tuple of fitted ``IsolationForest`` and binary predictions.
    """
    _, X, _ = _prepare_matrix(
        df,
        scale=scale,
        drop_weather=drop_weather,
        fit_indices=fit_indices,
        preprocessor=preprocessor,
    )

    if fit_indices is None:
        X_fit = X
    else:
        X_fit = X[fit_indices]

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
    preprocessor: AnomalyPreprocessor | None = None,
    fit_indices: np.ndarray | None = None,
    drop_weather: bool = False,
) -> tuple[DBSCAN, np.ndarray]:
    """Train DBSCAN on engineered smart meter features.

    Legacy callers use ``scale=False`` (default). Enhanced tuning sets
    ``scale=True`` with ``fit_indices`` for train-only scaler fit.

    Args:
        df: Feature-engineered DataFrame.
        eps: Neighborhood radius.
        min_samples: Minimum cluster size.
        metric: Distance metric (``euclidean`` or ``manhattan``).
        scale: Apply train-fitted ``StandardScaler`` (+ hourly z-score).
        preprocessor: Optional pre-fitted preprocessor.
        fit_indices: Rows used to fit the scaler (DBSCAN still sees all rows).
        drop_weather: Drop weather columns before clustering.

    Returns:
        Tuple of fitted ``DBSCAN`` and binary predictions.
    """
    _, X, _ = _prepare_matrix(
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
    """Combine Isolation Forest and DBSCAN predictions.

    Args:
        df: Feature-engineered DataFrame.
        if_kwargs: Forwarded to ``train_isolation_forest``.
        dbscan_kwargs: Forwarded to ``train_dbscan``.
        strategy: ``intersection``, ``union``, or ``weighted``.
        alpha: IF weight for ``weighted`` strategy (DBSCAN weight = 1 - alpha).
        score_threshold: IF threshold for weighted strategy (required unless
            provided in ``if_kwargs``).
        fit_indices: Chronological train rows for scaler/model fit.
        drop_weather: Drop weather columns.

    Returns:
        Dictionary with component models and binary ensemble predictions.
    """
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
        feature_matrix, X, _ = _prepare_matrix(
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
    """Route feature-engineered data to an unsupervised anomaly detector.

    Unified entry point for Isolation Forest, DBSCAN, and ensemble mode.
    Production defaults use legacy 15-column features with no scaling.

    Args:
        df: Raw or feature-engineered DataFrame.
        model_type: ``isolation_forest``, ``dbscan``, or ``ensemble``.
        feature_builder: Optional builder applied when ``df`` lacks engineered
            columns (defaults to ``build_all_features`` when ``hour`` missing).
        **kwargs: Hyperparameters forwarded to the selected trainer.

    Returns:
        Tuple of fitted model (or bundle for ensemble) and binary predictions.
    """
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

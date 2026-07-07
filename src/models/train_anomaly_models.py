"""Unsupervised anomaly detection model training for Phase 2.

Isolation Forest and DBSCAN train on engineered features only.
``Anomaly_Label`` is excluded before fitting and is used strictly as an
evaluation benchmark — see docs/phase2-strategy.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

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


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[IsolationForest, np.ndarray]:
    """Train an Isolation Forest on engineered smart meter features.

    ``Anomaly_Label`` is never passed to the model. Predictions use the
    evaluation encoding: ``0`` = Normal, ``1`` = Abnormal.

    Args:
        df: Feature-engineered DataFrame (temporal + rolling columns applied).
        contamination: Expected proportion of outliers in the data (default 0.05
            matches the ~5% Abnormal benchmark rate).
        random_state: Random seed for reproducible training.

    Returns:
        Tuple of the fitted ``IsolationForest`` and binary predictions
        (``0`` = Normal, ``1`` = Abnormal). Prediction length equals the
        number of rows after NaN warm-up rows are dropped.

    Raises:
        ValueError: If no rows remain after feature preparation.
    """
    feature_matrix = prepare_feature_matrix(df.copy())

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(feature_matrix)

    raw_predictions = model.predict(feature_matrix)
    predictions = (raw_predictions == -1).astype(int)
    return model, predictions


def train_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[DBSCAN, np.ndarray]:
    """Train DBSCAN on engineered smart meter features.

    ``Anomaly_Label`` is never passed to the model. DBSCAN assigns noise
    points (label ``-1``) to sparse regions; cluster members (label ``>= 0``)
    are treated as normal baseline density. Predictions use the evaluation
    encoding: ``0`` = Normal, ``1`` = Abnormal.

    Args:
        df: Feature-engineered DataFrame (temporal + rolling columns applied).
        eps: Maximum distance between two samples for neighborhood membership.
        min_samples: Minimum points required to form a dense region.

    Returns:
        Tuple of the fitted ``DBSCAN`` and binary predictions
        (``0`` = Normal, ``1`` = Abnormal). Prediction length equals the
        number of rows after NaN warm-up rows are dropped.

    Raises:
        ValueError: If no rows remain after feature preparation.
    """
    feature_matrix = prepare_feature_matrix(df.copy())

    model = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = model.fit_predict(feature_matrix)
    predictions = (raw_labels == -1).astype(int)
    return model, predictions


def detect_anomalies(
    df: pd.DataFrame,
    model_type: str = "isolation_forest",
    **kwargs: object,
) -> tuple[IsolationForest | DBSCAN, np.ndarray]:
    """Route feature-engineered data to an unsupervised anomaly detector.

    Unified entry point for Isolation Forest and DBSCAN. ``Anomaly_Label`` is
    never used for training (handled inside each trainer). Predictions use the
    evaluation encoding: ``0`` = Normal, ``1`` = Abnormal.

    Args:
        df: Feature-engineered DataFrame (temporal + rolling columns applied).
        model_type: ``"isolation_forest"`` (default) or ``"dbscan"``.
        **kwargs: Hyperparameters forwarded to the selected trainer
            (e.g. ``contamination``, ``random_state`` for Isolation Forest;
            ``eps``, ``min_samples`` for DBSCAN).

    Returns:
        Tuple of the fitted model and binary predictions. Prediction length
        equals the number of rows after NaN warm-up rows are dropped.

    Raises:
        ValueError: If ``model_type`` is not supported.
    """
    model_key = model_type.strip().lower()
    if model_key == "isolation_forest":
        return train_isolation_forest(df, **kwargs)
    if model_key == "dbscan":
        return train_dbscan(df, **kwargs)
    raise ValueError(
        f"Unsupported model_type {model_type!r}. "
        "Use 'isolation_forest' or 'dbscan'."
    )

"""Feature matrix preparation for unsupervised anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.anomaly_preprocessing import AnomalyPreprocessor

EXCLUDE_COLUMNS = frozenset({"Timestamp", "Anomaly_Label"})
REQUIRED_CONTEXT_COLUMNS = frozenset({"hour", "Electricity_Consumed"})
WEATHER_COLUMNS = ("Temperature", "Humidity", "Wind_Speed")


def apply_feature_ablation(
    feature_matrix: pd.DataFrame,
    drop_weather: bool = True,
) -> pd.DataFrame:
    """Drop weather columns for ablation experiments."""
    if not drop_weather:
        return feature_matrix
    drop_cols = [c for c in WEATHER_COLUMNS if c in feature_matrix.columns]
    return feature_matrix.drop(columns=drop_cols)


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


def aligned_context(
    df: pd.DataFrame, feature_matrix: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Return hour and consumption series aligned to ``feature_matrix`` rows."""
    missing = REQUIRED_CONTEXT_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns required for preprocessing: {sorted(missing)}. "
            "Run feature engineering before preparing the matrix."
        )
    index = feature_matrix.index
    hours = df.loc[index, "hour"]
    consumption = df.loc[index, "Electricity_Consumed"]
    return hours, consumption


def prepare_model_matrix(
    df: pd.DataFrame,
    *,
    scale: bool,
    drop_weather: bool,
    fit_indices: np.ndarray | None,
    preprocessor: AnomalyPreprocessor | None,
) -> tuple[pd.DataFrame, np.ndarray, AnomalyPreprocessor | None]:
    """Build numeric matrix with optional train-fitted scaling."""
    feature_matrix = apply_feature_ablation(prepare_feature_matrix(df.copy()), drop_weather)
    hours, consumption = aligned_context(df, feature_matrix)

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

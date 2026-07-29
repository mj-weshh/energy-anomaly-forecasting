"""Feature engineering for the Smart Meter Electricity Consumption Dataset.

This module is the canonical home for all Phase 2 feature engineering. It
transforms the validated DataFrame produced by :mod:`src.data.ingest_data`
into a model-ready feature matrix for unsupervised anomaly detection
(Isolation Forest, DBSCAN) and, later, Phase 3 forecasting.

Phase 1 EDA (docs/eda-insights.md) showed that consumption depends strongly
on time of day (peak mean load at 02:00) and, more subtly, on day of week.
Context-aware detectors therefore need explicit temporal features rather
than raw timestamps.

Public API (Phase 2, Week 3):

- ``add_temporal_features(df)`` — hour, day-of-week, month, weekend flag
- ``add_rolling_metrics(df)`` — 3-hour and 24-hour rolling mean / standard
  deviation over consumption
- ``build_all_features(df)`` — apply temporal then rolling features in one call

Public API (Phase 3, Week 7 — XGBoost prep):

- ``create_supervised_lags(df, target_col)`` — lag columns at t-1, t-2, t-48
  for supervised forecasting

Public API (Phase 3, Week 7 Day 3 — LSTM prep):

- ``create_sequences(data, seq_length)`` — sliding-window 3D arrays for LSTM
  input ``[samples, time_steps, features]``

Usage:
    Import in downstream scripts and notebooks::

        from src.features.build_features import add_temporal_features

    Always load data through the canonical ingestion module first::

        from src.data.ingest_data import find_dataset_csv, load_smart_meter_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive basic temporal context columns from ``Timestamp``.

    Args:
        df: DataFrame with a parsed ``Timestamp`` column
            (as produced by ``src.data.ingest_data.load_smart_meter_data``).

    Returns:
        Copy of ``df`` with added integer columns:
        ``hour`` (0-23), ``day_of_week`` (0=Monday .. 6=Sunday),
        ``month`` (1-12), and ``is_weekend`` (1 if Saturday/Sunday else 0).

    Raises:
        KeyError: If ``Timestamp`` is not present in ``df``.
    """
    if "Timestamp" not in df.columns:
        raise KeyError("Required column 'Timestamp' not found in DataFrame.")

    df = df.copy()
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rolling consumption statistics for local-context anomaly scoring.

    Rolling windows are only meaningful on chronologically ordered data, so
    rows are sorted by ``Timestamp`` before any window math.

    Args:
        df: DataFrame with parsed ``Timestamp`` and ``Electricity_Consumed``
            columns (as produced by ``src.data.ingest_data``).

    Returns:
        Chronologically sorted copy of ``df`` with added columns over
        ``Electricity_Consumed``: ``rolling_mean_3h`` and ``rolling_std_3h``
        (3-hour window) plus ``rolling_mean_24h`` and ``rolling_std_24h``
        (24-hour window). For each window the first ``window - 1`` rows are
        NaN until the window fills.

    Raises:
        KeyError: If ``Timestamp`` or ``Electricity_Consumed`` is not
            present in ``df``.
    """
    for column in ("Timestamp", "Electricity_Consumed"):
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' not found in DataFrame.")

    df = df.sort_values("Timestamp").copy()

    window_3h = 6  # 3 hours at 30-minute intervals
    rolling_3h = df["Electricity_Consumed"].rolling(window=window_3h)
    df["rolling_mean_3h"] = rolling_3h.mean()
    df["rolling_std_3h"] = rolling_3h.std()

    window_24h = 48  # 24 hours at 30-minute intervals
    rolling_24h = df["Electricity_Consumed"].rolling(window=window_24h)
    df["rolling_mean_24h"] = rolling_24h.mean()
    df["rolling_std_24h"] = rolling_24h.std()
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply temporal features then rolling metrics.

    Convenience wrapper for the full Phase 2 Week 3 feature pipeline.

    Args:
        df: Validated DataFrame from ``src.data.ingest_data``.

    Returns:
        Copy of ``df`` with temporal and rolling feature columns added.
    """
    return add_rolling_metrics(add_temporal_features(df))


def create_supervised_lags(
    df: pd.DataFrame,
    target_col: str = "Electricity_Consumed",
) -> pd.DataFrame:
    """Convert a time series into a supervised tabular frame with lag predictors.

    XGBoost and other tree models do not read time natively. Each row keeps the
    target at time ``t`` alongside past values at ``t-1`` (30 minutes),
    ``t-2`` (1 hour), and ``t-48`` (24 hours at 30-minute resolution).

    Rows are sorted by ``Timestamp`` when present. After shifting, rows with
    missing lag history are dropped so every remaining row has complete
    predictors. On a continuous 5,000-row clean series this removes the first
    **48** rows (the longest lag window). XGBoost can tolerate NaNs, but we
    drop them so all forecast models evaluate on the same timeline.

    Args:
        df: Chronological smart-meter DataFrame containing ``target_col``.
        target_col: Column to lag. Defaults to ``Electricity_Consumed``.

    Returns:
        Copy of ``df`` with added columns ``{target_col}_lag_1``,
        ``{target_col}_lag_2``, and ``{target_col}_lag_48``. Rows with NaN
        lag values are removed and the index is reset.

    Raises:
        KeyError: If ``target_col`` is missing from ``df``.
    """
    if target_col not in df.columns:
        raise KeyError(f"Required column '{target_col}' not found in DataFrame.")

    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").copy()
    else:
        df = df.copy()

    lag_1 = f"{target_col}_lag_1"
    lag_2 = f"{target_col}_lag_2"
    lag_48 = f"{target_col}_lag_48"

    df[lag_1] = df[target_col].shift(1)
    df[lag_2] = df[target_col].shift(2)
    df[lag_48] = df[target_col].shift(48)

    return df.dropna(subset=[lag_1, lag_2, lag_48]).reset_index(drop=True)


def create_sequences(
    data: np.ndarray,
    seq_length: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for LSTM-style sequence models.

    LSTMs expect 3D input tensors ``[samples, time_steps, features]``. This
    helper slides a fixed-length window over a chronological 2D feature matrix
    and pairs each window with the **next** row as the prediction target.

    At 30-minute resolution, ``seq_length=24`` corresponds to **12 hours**
    of history per sample.

    Args:
        data: 2D array of shape ``(n_timesteps, n_features)`` in time order.
        seq_length: Number of past timesteps per sample. Defaults to ``24``.

    Returns:
        Tuple ``(X, y)`` where:

        - ``X`` has shape ``(num_samples, seq_length, n_features)``
        - ``y`` has shape ``(num_samples, n_features)``

        with ``num_samples = n_timesteps - seq_length``.

    Raises:
        ValueError: If ``data`` is not 2D, ``seq_length`` is less than 1, or
            there are not enough timesteps to form at least one window.
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be a 2D array (n_timesteps, n_features); got ndim={data.ndim}."
        )
    if seq_length < 1:
        raise ValueError(f"seq_length must be at least 1; got {seq_length}.")
    if len(data) <= seq_length:
        raise ValueError(
            f"Need more than {seq_length} timesteps to build sequences; got {len(data)}."
        )

    n_timesteps, n_features = data.shape
    num_samples = n_timesteps - seq_length

    X = np.empty((num_samples, seq_length, n_features), dtype=data.dtype)
    y = np.empty((num_samples, n_features), dtype=data.dtype)

    for i in range(num_samples):
        X[i] = data[i : i + seq_length]
        y[i] = data[i + seq_length]

    return X, y


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour and day-of-week as sin/cos pairs for distance-based models.

    Args:
        df: DataFrame with ``hour`` and ``day_of_week`` columns (from
            ``add_temporal_features``).

    Returns:
        Copy of ``df`` with ``hour_sin``, ``hour_cos``, ``dow_sin``, ``dow_cos``.

    Raises:
        KeyError: If required temporal columns are missing.
    """
    for column in ("hour", "day_of_week"):
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' not found in DataFrame.")

    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_consumption_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Add step change and local-baseline residual features for consumption.

    Args:
        df: Chronologically sorted DataFrame with ``Electricity_Consumed`` and
            ``rolling_mean_24h`` (from ``add_rolling_metrics``).

    Returns:
        Copy of ``df`` with ``consumption_diff`` and ``consumption_residual_24h``.

    Raises:
        KeyError: If required columns are missing.
    """
    for column in ("Electricity_Consumed", "rolling_mean_24h"):
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' not found in DataFrame.")

    df = df.sort_values("Timestamp").copy() if "Timestamp" in df.columns else df.copy()
    df["consumption_diff"] = df["Electricity_Consumed"].diff()
    df["consumption_residual_24h"] = (
        df["Electricity_Consumed"] - df["rolling_mean_24h"]
    )
    return df


def build_enhanced_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply legacy features plus cyclical encoding and consumption derivatives.

    Research/tuning pipeline entry point. Production clean-data still uses
    ``build_all_features`` (15 columns).

    Args:
        df: Validated DataFrame from ``src.data.ingest_data``.

    Returns:
        Copy of ``df`` with 21 feature-engineered columns (7 original + 14 new).
    """
    featured = build_all_features(df)
    return add_consumption_derivatives(add_cyclical_features(featured))

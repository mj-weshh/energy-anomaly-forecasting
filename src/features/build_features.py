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

Usage:
    Import in downstream scripts and notebooks::

        from src.features.build_features import add_temporal_features

    Always load data through the canonical ingestion module first::

        from src.data.ingest_data import find_dataset_csv, load_smart_meter_data
"""

from __future__ import annotations

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

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
- Rolling statistics (local mean / standard deviation) planned for a later step

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

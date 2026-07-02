"""Feature engineering for the Smart Meter Electricity Consumption Dataset.

This module is the canonical home for all Phase 2 feature engineering. It
transforms the validated DataFrame produced by :mod:`src.data.ingest_data`
into a model-ready feature matrix for unsupervised anomaly detection
(Isolation Forest, DBSCAN) and, later, Phase 3 forecasting.

Phase 1 EDA (docs/eda-insights.md) showed that consumption depends strongly
on time of day (peak mean load at 02:00) and, more subtly, on day of week.
Context-aware detectors therefore need explicit temporal features rather
than raw timestamps.

Planned public API (implemented incrementally during Phase 2, Week 3):

- ``add_temporal_features(df)`` — hour, day-of-week, month, weekend flag
- Rolling statistics (local mean / standard deviation) in a later step

Usage:
    Import in downstream scripts and notebooks::

        from src.features.build_features import add_temporal_features

    Always load data through the canonical ingestion module first::

        from src.data.ingest_data import find_dataset_csv, load_smart_meter_data
"""

from __future__ import annotations

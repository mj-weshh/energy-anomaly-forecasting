"""Feature engineering utilities for anomaly detection and forecasting.

Public API is defined in :mod:`src.features.build_features`. Import directly::

    from src.features.build_features import add_temporal_features

Functions will be added incrementally during Phase 2 (temporal features,
rolling statistics) and re-exported here once stable.
"""

__all__ = [
    "add_consumption_derivatives",
    "add_cyclical_features",
    "add_rolling_metrics",
    "add_temporal_features",
    "build_all_features",
    "build_enhanced_anomaly_features",
]

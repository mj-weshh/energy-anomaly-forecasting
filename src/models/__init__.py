"""Unsupervised anomaly detection models and evaluation utilities.

Public API is defined in submodules. Import directly::

    from src.models.evaluate_models import evaluate_anomaly_model
    from src.models.train_anomaly_models import (
        prepare_feature_matrix,
        train_isolation_forest,
    )
"""

__all__ = [
    "evaluate_anomaly_model",
    "prepare_feature_matrix",
    "train_isolation_forest",
]

"""Research-optimized anomaly detection defaults from temporal tuning.

These constants are populated by ``scripts/tune_*.py`` and are **not** used
by the production clean-data pipeline (``generate_clean_dataset``), which
keeps legacy 15-column features and default Isolation Forest params.

Re-run tuning scripts after feature or split changes to refresh values.
"""

from __future__ import annotations

# Best validation configs from scripts/tune_isolation_forest.py (2026-07-14 run).
BEST_IF_CONFIG: dict = {
    "contamination": 0.03,
    "n_estimators": 200,
    "max_features": 0.6,
    "scale": True,
    "drop_weather": False,
    "use_score_threshold": True,
    "score_threshold": 0.016764,
}

# Best validation config from scripts/tune_dbscan.py (enhanced, scaled).
BEST_DBSCAN_CONFIG: dict = {
    "eps": 10.0,
    "min_samples": 10,
    "metric": "manhattan",
    "scale": True,
    "drop_weather": False,
}

# Best validation config from scripts/tune_ensemble.py (aligned with anomaly_config IF/DBSCAN).
BEST_ENSEMBLE_CONFIG: dict = {
    "strategy": "union",
    "alpha": 0.7,
}

# Measured F1 on temporal test split (60/20/20 chronological).
TUNING_METRICS: dict = {
    "legacy_if_full_dataset_f1": 0.331,
    "enhanced_if_test_f1": 0.460,
    "enhanced_if_val_f1": 0.611,
    "enhanced_dbscan_test_f1": 0.297,
    "enhanced_dbscan_val_f1": 0.509,
    "enhanced_ensemble_union_test_f1": 0.400,
    "enhanced_ensemble_union_val_f1": 0.565,
    "enhanced_ensemble_intersection_test_f1": 0.329,
}

TEMPORAL_SPLIT_RATIOS: dict = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2,
}

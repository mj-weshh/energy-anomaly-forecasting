"""Research-optimized anomaly detection defaults from temporal tuning.

These constants are populated by ``scripts/tune_*.py`` and are **not** used
by the production clean-data pipeline (``generate_clean_dataset``), which
keeps legacy 15-column features and default Isolation Forest params.

Re-run tuning scripts after feature or split changes to refresh values.
"""

from __future__ import annotations

from typing import TypedDict

# Best validation configs from scripts/tune_isolation_forest.py (2026-07-14 run).


class IFConfig(TypedDict):
    contamination: float
    n_estimators: int
    max_features: float
    scale: bool
    drop_weather: bool
    use_score_threshold: bool
    score_threshold: float


class DBSCANConfig(TypedDict):
    eps: float
    min_samples: int
    metric: str
    scale: bool
    drop_weather: bool


class EnsembleConfig(TypedDict):
    strategy: str
    alpha: float


class TuningMetrics(TypedDict):
    legacy_if_full_dataset_f1: float
    legacy_if_test_f1: float
    legacy_if_test_precision: float
    legacy_if_test_recall: float
    legacy_if_test_val_threshold_f1: float
    legacy_if_test_val_threshold_precision: float
    legacy_if_test_val_threshold_recall: float
    enhanced_if_test_f1: float
    enhanced_if_test_precision: float
    enhanced_if_test_recall: float
    enhanced_if_val_f1: float
    enhanced_if_no_weather_test_f1: float
    enhanced_if_no_weather_test_precision: float
    enhanced_if_no_weather_test_recall: float
    enhanced_dbscan_test_f1: float
    enhanced_dbscan_val_f1: float
    enhanced_ensemble_union_test_f1: float
    enhanced_ensemble_union_val_f1: float
    enhanced_ensemble_intersection_test_f1: float


BEST_IF_CONFIG: IFConfig = {
    "contamination": 0.03,
    "n_estimators": 200,
    "max_features": 0.6,
    "scale": True,
    "drop_weather": False,
    "use_score_threshold": True,
    "score_threshold": 0.016764,
}

# Weather ablation from scripts/tune_isolation_forest.py --drop-weather (2026-07-15).
BEST_IF_CONFIG_NO_WEATHER: IFConfig = {
    "contamination": 0.03,
    "n_estimators": 200,
    "max_features": 1.0,
    "scale": True,
    "drop_weather": True,
    "use_score_threshold": True,
    "score_threshold": 0.023322,
}

# Best validation config from scripts/tune_dbscan.py (enhanced, scaled).
BEST_DBSCAN_CONFIG: DBSCANConfig = {
    "eps": 10.0,
    "min_samples": 10,
    "metric": "manhattan",
    "scale": True,
    "drop_weather": False,
}

# Best validation config from scripts/tune_ensemble.py (aligned with anomaly_config IF/DBSCAN).
BEST_ENSEMBLE_CONFIG: EnsembleConfig = {
    "strategy": "union",
    "alpha": 0.7,
}

# Measured F1 on temporal test split (60/20/20 chronological).
TUNING_METRICS: TuningMetrics = {
    "legacy_if_full_dataset_f1": 0.331,
    "legacy_if_test_f1": 0.340,
    "legacy_if_test_precision": 0.214,
    "legacy_if_test_recall": 0.818,
    "legacy_if_test_val_threshold_f1": 0.389,
    "legacy_if_test_val_threshold_precision": 0.379,
    "legacy_if_test_val_threshold_recall": 0.400,
    "enhanced_if_test_f1": 0.460,
    "enhanced_if_test_precision": 0.625,
    "enhanced_if_test_recall": 0.364,
    "enhanced_if_val_f1": 0.611,
    "enhanced_if_no_weather_test_f1": 0.524,
    "enhanced_if_no_weather_test_precision": 0.562,
    "enhanced_if_no_weather_test_recall": 0.491,
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

# From scripts/compare_clean_artifacts.py first run (2026-07-15).
ARTIFACT_DIFF_METRICS: dict[str, float | int] = {
    "legacy_imputed_rows": 248,
    "legacy_threshold_imputed_rows": 178,
    "enhanced_imputed_rows": 51,
    "legacy_vs_legacy_threshold_jaccard": 0.257,
    "legacy_vs_enhanced_jaccard": 0.154,
    "legacy_threshold_vs_enhanced_jaccard": 0.106,
    "legacy_vs_enhanced_disagree_rows": 219,
    "top_disagree_hour": 0,
}

"""Orchestration for the Phase 3 clean dataset artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.ensemble import IsolationForest

from src.data.ingest_data import load_smart_meter_data
from src.features.build_features import build_all_features, build_enhanced_anomaly_features
from src.models.anomaly_config import BEST_IF_CONFIG
from src.models.feature_matrix import prepare_feature_matrix
from src.models.train_anomaly_models import train_isolation_forest
from src.models.tuning_utils import (
    find_best_threshold,
    isolation_forest_scores,
    predict_from_scores,
    temporal_train_val_test_split,
)

CleanProfile = Literal["legacy", "legacy_threshold", "enhanced"]

DEFAULT_OUTPUTS: dict[CleanProfile, str] = {
    "legacy": "clean_smart_meter_data.csv",
    "legacy_threshold": "clean_smart_meter_data_legacy_threshold.csv",
    "enhanced": "clean_smart_meter_data_enhanced.csv",
}


def _legacy_threshold_predictions(df_feat) -> np.ndarray:
    """Train legacy IF on 60% train; threshold on 20% val; score all eval rows."""
    from src.models.tuning_utils import align_labels

    feature_matrix = prepare_feature_matrix(df_feat)
    X = feature_matrix.to_numpy(dtype=float)
    train_idx, val_idx, _ = temporal_train_val_test_split(len(feature_matrix))

    model = IsolationForest(
        contamination=0.05,
        n_estimators=100,
        max_features=1.0,
        random_state=42,
    )
    model.fit(X[train_idx])
    y_val = align_labels(df_feat, feature_matrix.index)[val_idx]
    val_scores = isolation_forest_scores(model, X[val_idx])
    threshold, _ = find_best_threshold(val_scores, y_val)
    return predict_from_scores(isolation_forest_scores(model, X), threshold)


def _enhanced_predictions(df_feat) -> np.ndarray:
    """Enhanced IF on all eval rows with tuned config and score threshold."""
    _, predictions = train_isolation_forest(
        df_feat,
        contamination=BEST_IF_CONFIG["contamination"],
        n_estimators=BEST_IF_CONFIG["n_estimators"],
        max_features=BEST_IF_CONFIG["max_features"],
        scale=BEST_IF_CONFIG["scale"],
        drop_weather=BEST_IF_CONFIG["drop_weather"],
        score_threshold=BEST_IF_CONFIG["score_threshold"],
    )
    return predictions


def generate_clean_dataset(
    input_path: str,
    output_path: str,
    *,
    profile: CleanProfile = "legacy",
) -> Path:
    """Build a continuity-safe clean dataset for Phase 3 forecasting.

    End-to-end pipeline: load raw CSV → engineer features → detect anomalies
    → mask and time-interpolate anomalous ``Electricity_Consumed`` values →
    save to ``output_path``.

    Row count is preserved (5000 rows on the canonical dataset). Output
    includes feature-engineered columns with imputed consumption.

    Args:
        input_path: Path to raw ``smart_meter_data.csv``.
        output_path: Destination path for the cleaned CSV.
        profile: Detection policy — ``legacy`` (default, unchanged),
            ``legacy_threshold`` (val-tuned threshold, legacy features),
            or ``enhanced`` (research IF config on enhanced features).

    Returns:
        Resolved absolute path to the written CSV file.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        KeyError: If required columns are missing during processing.
        ValueError: If anomaly predictions cannot be aligned or validated.
    """
    input_csv = Path(input_path)
    output_csv = Path(output_path)

    df = load_smart_meter_data(input_csv)

    if profile == "legacy":
        df_feat = build_all_features(df)
        _, predictions = train_isolation_forest(df_feat)
    elif profile == "legacy_threshold":
        df_feat = build_all_features(df)
        predictions = _legacy_threshold_predictions(df_feat)
    elif profile == "enhanced":
        df_feat = build_enhanced_anomaly_features(df)
        predictions = _enhanced_predictions(df_feat)
    else:
        raise ValueError(f"Unsupported profile: {profile!r}")

    from src.data.clean_data import interpolate_anomalies

    clean_df = interpolate_anomalies(df_feat, predictions)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    return output_csv.resolve()

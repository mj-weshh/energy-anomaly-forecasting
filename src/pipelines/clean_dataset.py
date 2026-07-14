"""Orchestration for the Phase 3 clean dataset artifact."""

from __future__ import annotations

from pathlib import Path

from src.data.clean_data import interpolate_anomalies
from src.data.ingest_data import load_smart_meter_data
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import detect_anomalies


def generate_clean_dataset(input_path: str, output_path: str) -> Path:
    """Build a continuity-safe clean dataset for Phase 3 forecasting.

    End-to-end pipeline: load raw CSV → engineer features → detect anomalies
    with Isolation Forest (default) → mask and time-interpolate anomalous
    ``Electricity_Consumed`` values → save to ``output_path``.

    Row count is preserved (5000 rows on the canonical dataset). Output
    includes feature-engineered columns with imputed consumption.

    Args:
        input_path: Path to raw ``smart_meter_data.csv``.
        output_path: Destination path for the cleaned CSV.

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
    df_feat = build_all_features(df)
    _, predictions = detect_anomalies(df_feat, model_type="isolation_forest")
    clean_df = interpolate_anomalies(df_feat, predictions)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    return output_csv.resolve()

"""End-to-end CLI entry point for the energy anomaly + forecasting pipeline.

This module is the single user-facing command for running the Phase 1–3
workflow from the repository root. Later steps wire ingestion, feature
engineering, cleaning, and model training via ``src/`` packages.

Example:
    python main.py
    python main.py --model lstm --epochs 30
    python main.py --data_path path/to/smart_meter_data.csv
    python main.py --save_clean_data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.clean_data import interpolate_anomalies
from src.data.ingest_data import load_smart_meter_data
from src.data.make_forecast_dataset import time_series_split
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import detect_anomalies

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the E2E pipeline.

    Returns:
        Parsed namespace with ``data_path``, ``model``, ``epochs``, and
        ``save_clean_data``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the energy anomaly detection and forecasting end-to-end "
            "pipeline from raw smart-meter CSV through selected forecaster."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("Smart Meter Electricity Consumption Dataset")
        / "smart_meter_data.csv",
        help=(
            "Path to the smart meter CSV. Defaults to the bundled Kaggle "
            "dataset path under the repository root."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "prophet", "xgboost", "lstm"],
        default="naive",
        help=(
            "Forecasting model to run after data preparation. "
            "Choices: naive, prophet, xgboost, lstm. Default: naive."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs when --model is lstm. Default: 20.",
    )
    parser.add_argument(
        "--save_clean_data",
        action="store_true",
        help=(
            "Save interpolated clean dataframe to "
            "data/processed/clean_pipeline_output.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run the E2E pipeline.

    Day 2 wires Phase 1 ingestion and Phase 2 feature engineering.
    Day 3 adds Isolation Forest detection, in-memory interpolation, and
    optional ``--save_clean_data``. Day 4 Step 1 adds chronological
    train/val/test splitting; model routing follows in later steps.
    """
    args = parse_args()
    logger.info(
        "E2E pipeline starting (model=%s, epochs=%s, data_path=%s, "
        "save_clean_data=%s)",
        args.model,
        args.epochs,
        args.data_path,
        args.save_clean_data,
    )

    data_path = Path(args.data_path)
    logger.info("Starting data ingestion from %s ...", data_path)
    df = load_smart_meter_data(data_path)
    logger.info("Raw data loaded: shape=%s", df.shape)

    logger.info("Building temporal and rolling features ...")
    df_feat = build_all_features(df)
    warmup_nans = int(df_feat.isna().any(axis=1).sum())
    logger.info(
        "Feature matrix ready: shape=%s (%s rows with rolling-window warm-up NaNs; "
        "row count preserved - no rows dropped)",
        df_feat.shape,
        warmup_nans,
    )

    logger.info("Running Isolation Forest anomaly detection ...")
    _model, predictions = detect_anomalies(df_feat, model_type="isolation_forest")
    n_anomalies = int(predictions.sum())
    logger.info(
        "Anomalies detected: %s of %s scored rows",
        n_anomalies,
        len(predictions),
    )

    logger.info("Masking anomalies and time-interpolating Electricity_Consumed ...")
    df_clean = interpolate_anomalies(df_feat, predictions)
    logger.info(
        "Clean in-memory dataset ready: shape=%s, consumption_NaNs=%s",
        df_clean.shape,
        int(df_clean["Electricity_Consumed"].isna().sum()),
    )

    if args.save_clean_data:
        out_path = Path("data/processed/clean_pipeline_output.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)
        logger.info("Saved clean pipeline output to %s", out_path.resolve())

    logger.info("Chronological train/val/test split (70/15/15) ...")
    train_df, val_df, test_df = time_series_split(df_clean)
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        logger.info(
            "%s split: rows=%s, %s -> %s",
            name,
            len(frame),
            frame["Timestamp"].iloc[0],
            frame["Timestamp"].iloc[-1],
        )
    # Later steps: --model routing and training.


if __name__ == "__main__":
    main()

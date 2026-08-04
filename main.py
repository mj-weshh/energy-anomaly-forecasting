"""End-to-end CLI entry point for the energy anomaly + forecasting pipeline.

This module is the single user-facing command for running the Phase 1–3
workflow from the repository root. Later steps wire ingestion, feature
engineering, cleaning, and model training via ``src/`` packages.

Example:
    python main.py
    python main.py --model lstm --epochs 30
    python main.py --data_path path/to/smart_meter_data.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the E2E pipeline.

    Returns:
        Parsed namespace with ``data_path``, ``model``, and ``epochs``.
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
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run the E2E pipeline.

    Logging is configured at module load. Subsequent Day 2 steps wire
    ingestion and feature engineering.
    """
    args = parse_args()
    logger.info(
        "E2E pipeline starting (model=%s, epochs=%s, data_path=%s)",
        args.model,
        args.epochs,
        args.data_path,
    )
    # Pipeline body (ingestion, features, models) is wired in later steps.


if __name__ == "__main__":
    main()

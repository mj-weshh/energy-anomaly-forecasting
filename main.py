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

import numpy as np
import pandas as pd

from src.data.clean_data import interpolate_anomalies
from src.data.ingest_data import load_smart_meter_data
from src.data.make_forecast_dataset import time_series_split
from src.features.build_features import (
    build_all_features,
    create_sequences,
    create_supervised_lags,
)
from src.models.lstm_model import EnergyLSTM
from src.models.train_anomaly_models import detect_anomalies
from src.models.train_forecast_models import (
    make_lstm_dataloader,
    naive_seasonal_forecast,
    predict_lstm,
    train_lstm_model,
    train_prophet_model,
    train_xgboost_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "Electricity_Consumed"
SEASONAL_PERIODS = 48
TRAIN_PCT = 0.7
VAL_PCT = 0.15
SEQ_LENGTH = 24
BATCH_SIZE = 32

XGBOOST_FEATURE_COLUMNS = [
    f"{TARGET_COLUMN}_lag_1",
    f"{TARGET_COLUMN}_lag_2",
    f"{TARGET_COLUMN}_lag_48",
    "Temperature",
    "Humidity",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

LSTM_FEATURE_COLUMNS = [
    TARGET_COLUMN,
    "Temperature",
    "Humidity",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]


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


def _split_sequence_arrays(
    X: np.ndarray,
    y: np.ndarray,
    train_pct: float = TRAIN_PCT,
    val_pct: float = VAL_PCT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split sequence arrays using 70/15/15 fraction math."""
    n = len(X)
    if n == 0:
        raise ValueError("Cannot split empty sequence arrays.")
    if len(y) != n:
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError(
            f"One or more splits are empty after slicing; n={n}, "
            f"train_end={train_end}, val_end={val_end}."
        )
    return (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )


def run_selected_forecast(
    model_name: str,
    df_clean: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
) -> np.ndarray:
    """Train/predict with the CLI-selected forecaster using native prep.

    Args:
        model_name: One of ``naive``, ``prophet``, ``xgboost``, ``lstm``.
        df_clean: Full cleaned timeline (needed for XGBoost/LSTM warm-up).
        train_df: Chronological train split of ``df_clean``.
        val_df: Chronological validation split (unused by naive/Prophet).
        test_df: Chronological test split of ``df_clean``.
        epochs: LSTM training epochs from ``--epochs``.

    Returns:
        1-D NumPy array of test-window predictions.
    """
    del val_df  # reserved for models that monitor validation during fit

    if model_name == "naive":
        logger.info("Training forecast model: naive (seasonal persistence) ...")
        y_pred = naive_seasonal_forecast(
            train_df[TARGET_COLUMN],
            test_df[TARGET_COLUMN],
            seasonal_periods=SEASONAL_PERIODS,
        )
        return np.asarray(y_pred, dtype=float)

    if model_name == "prophet":
        logger.info("Training forecast model: prophet ...")
        y_pred = train_prophet_model(train_df, test_df)
        return np.asarray(y_pred, dtype=float)

    if model_name == "xgboost":
        logger.info("Training forecast model: xgboost (supervised lags) ...")
        tabular_df = create_supervised_lags(df_clean, target_col=TARGET_COLUMN)
        x_train, x_val, x_test = time_series_split(tabular_df)
        model = train_xgboost_model(
            x_train[XGBOOST_FEATURE_COLUMNS],
            x_train[TARGET_COLUMN],
            x_val[XGBOOST_FEATURE_COLUMNS],
            x_val[TARGET_COLUMN],
        )
        y_pred = model.predict(x_test[XGBOOST_FEATURE_COLUMNS])
        return np.asarray(y_pred, dtype=float)

    if model_name == "lstm":
        logger.info(
            "Training forecast model: lstm (seq_length=%s, epochs=%s) ...",
            SEQ_LENGTH,
            epochs,
        )
        data = df_clean[LSTM_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        X, y = create_sequences(data, seq_length=SEQ_LENGTH)
        X_train, X_val, X_test, y_train, y_val, y_test = _split_sequence_arrays(X, y)
        train_loader = make_lstm_dataloader(
            X_train, y_train, batch_size=BATCH_SIZE, shuffle=False
        )
        val_loader = make_lstm_dataloader(
            X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = make_lstm_dataloader(
            X_test, y_test, batch_size=BATCH_SIZE, shuffle=False
        )
        model = EnergyLSTM(input_size=len(LSTM_FEATURE_COLUMNS))
        model = train_lstm_model(model, train_loader, val_loader, epochs=epochs)
        y_pred = predict_lstm(model, test_loader)
        return np.asarray(y_pred, dtype=float)

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    """Parse CLI arguments and run the E2E pipeline.

    Days 2–3: ingest, features, Isolation Forest, interpolate, optional save.
    Day 4: chronological split and CLI-selected forecast training.
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

    y_pred = run_selected_forecast(
        args.model,
        df_clean,
        train_df,
        val_df,
        test_df,
        epochs=args.epochs,
    )
    logger.info(
        "Forecast complete for model=%s: prediction_length=%s",
        args.model,
        len(y_pred),
    )
    preview = np.asarray(y_pred[:5], dtype=float)
    logger.info(
        "Prediction preview (first %s values): %s",
        len(preview),
        np.array2string(preview, precision=6, separator=", "),
    )


if __name__ == "__main__":
    main()

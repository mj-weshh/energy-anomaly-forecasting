"""Run all Phase 3 forecasting models and collect test-set predictions.

Loads the Phase 2 clean CSV, executes Naive, Prophet, XGBoost, and LSTM on
their native chronological test pipelines, aggregates predictions, prints
a Markdown metrics table, and saves a 3-day comparison plot for documentation.

Run from repository root (use the project ``.venv``)::

    .venv\\Scripts\\activate
    python scripts/compare_forecasts.py

Or without activating::

    .venv\\Scripts\\python.exe scripts/compare_forecasts.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.data.make_forecast_dataset import time_series_split  # noqa: E402
from src.features.build_features import create_sequences, create_supervised_lags  # noqa: E402
from src.models.evaluate_forecast import evaluate_forecast  # noqa: E402
from src.models.lstm_model import EnergyLSTM  # noqa: E402
from src.models.train_forecast_models import (  # noqa: E402
    make_lstm_dataloader,
    naive_seasonal_forecast,
    predict_lstm,
    train_lstm_model,
    train_prophet_model,
    train_xgboost_model,
)

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
TARGET_COLUMN = "Electricity_Consumed"
SEASONAL_PERIODS = 48
TRAIN_PCT = 0.7
VAL_PCT = 0.15

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

SEQ_LENGTH = 24
BATCH_SIZE = 32
LSTM_EPOCHS = 20

MODEL_ORDER = ("naive", "prophet", "xgboost", "lstm")
MODEL_LABELS = {
    "naive": "Naive",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}

PLOT_WINDOW_STEPS = 144  # 3 days at 30-minute resolution
FORECAST_COMPARISON_PNG = REPO_ROOT / "docs" / "assets" / "forecast_comparison.png"


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps."""
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/compare_forecasts.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if TARGET_COLUMN not in df.columns:
        _fail(f"Column '{TARGET_COLUMN}' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("Timestamp", ascending=True).reset_index(drop=True)


def _ensure_prophet_available() -> None:
    if importlib.util.find_spec("prophet") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "Prophet is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/compare_forecasts.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/compare_forecasts.py"
    )


def _ensure_xgboost_available() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "XGBoost is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/compare_forecasts.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/compare_forecasts.py"
    )


def _ensure_torch_available() -> None:
    if importlib.util.find_spec("torch") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "PyTorch is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/compare_forecasts.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/compare_forecasts.py"
    )


def _forecast_frame(
    timestamps: pd.Series | np.ndarray,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build a standardized prediction frame for one model."""
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(y_true_arr)} vs {len(y_pred_arr)}."
        )
    if len(timestamps) != len(y_true_arr):
        raise ValueError(
            f"timestamps and y_true length mismatch: {len(timestamps)} vs {len(y_true_arr)}."
        )
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).reset_index(drop=True),
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
        }
    )


def split_sequence_arrays(
    X: np.ndarray,
    y: np.ndarray,
    train_pct: float = TRAIN_PCT,
    val_pct: float = VAL_PCT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split sequence arrays using ``time_series_split`` fraction math."""
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


def run_naive_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Seasonal naive predictions on the raw chronological test split."""
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    y_pred = naive_seasonal_forecast(
        y_train, y_test, seasonal_periods=SEASONAL_PERIODS
    )
    return _forecast_frame(test_df["Timestamp"], y_test, y_pred)


def run_prophet_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prophet predictions on the raw chronological test split."""
    y_test = test_df[TARGET_COLUMN]
    y_pred = train_prophet_model(train_df, test_df)
    return _forecast_frame(test_df["Timestamp"], y_test, y_pred)


def run_xgboost_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost predictions after supervised lag warm-up and chronological split."""
    tabular_df = create_supervised_lags(df, target_col=TARGET_COLUMN)

    missing_features = [
        col for col in XGBOOST_FEATURE_COLUMNS if col not in tabular_df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing XGBoost feature columns: {missing_features}")

    train_df, val_df, test_df = time_series_split(tabular_df)
    model = train_xgboost_model(
        train_df[XGBOOST_FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        val_df[XGBOOST_FEATURE_COLUMNS],
        val_df[TARGET_COLUMN],
    )
    y_pred = model.predict(test_df[XGBOOST_FEATURE_COLUMNS])
    return _forecast_frame(test_df["Timestamp"], test_df[TARGET_COLUMN], y_pred)


def run_lstm_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """LSTM predictions after sequence generation and chronological split."""
    missing_features = [
        col for col in LSTM_FEATURE_COLUMNS if col not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing LSTM feature columns: {missing_features}")

    data = df[LSTM_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X, y = create_sequences(data, seq_length=SEQ_LENGTH)
    sequence_timestamps = df["Timestamp"].iloc[SEQ_LENGTH:].reset_index(drop=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_sequence_arrays(X, y)
    n = len(X)
    val_end = int(n * (TRAIN_PCT + VAL_PCT))
    test_timestamps = sequence_timestamps.iloc[val_end:].reset_index(drop=True)

    train_loader = make_lstm_dataloader(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=False
    )
    val_loader = make_lstm_dataloader(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = make_lstm_dataloader(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False
    )

    print()
    print(f"Training EnergyLSTM for {LSTM_EPOCHS} epochs...")
    model = EnergyLSTM(input_size=len(LSTM_FEATURE_COLUMNS))
    model = train_lstm_model(model, train_loader, val_loader, epochs=LSTM_EPOCHS)
    y_pred = predict_lstm(model, test_loader)
    y_true = y_test[:, 0]

    return _forecast_frame(test_timestamps, y_true, y_pred)


def collect_all_predictions(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run all four forecasters and return standardized test prediction frames."""
    train_df, _val_df, test_df = time_series_split(df)

    print()
    print("Running naive seasonal baseline...")
    naive_df = run_naive_forecast(train_df, test_df)

    print("Running Prophet...")
    prophet_df = run_prophet_forecast(train_df, test_df)

    print("Running XGBoost...")
    xgboost_df = run_xgboost_forecast(df)

    print("Running LSTM...")
    lstm_df = run_lstm_forecast(df)

    return {
        "naive": naive_df,
        "prophet": prophet_df,
        "xgboost": xgboost_df,
        "lstm": lstm_df,
    }


def _print_prediction_summary(predictions: dict[str, pd.DataFrame]) -> None:
    print()
    print("Collected test-set predictions:")
    for name, frame in predictions.items():
        start = frame["timestamp"].min()
        end = frame["timestamp"].max()
        print(f"  {name:8s}: {len(frame):4d} rows  ({start} -> {end})")


def compute_metrics(
    predictions: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """Score each model's test predictions with MAE, RMSE, and MAPE."""
    results: dict[str, dict[str, float]] = {}
    for name, frame in predictions.items():
        results[name] = evaluate_forecast(frame["y_true"], frame["y_pred"])
    return results


def print_markdown_table(results_dict: dict[str, dict[str, float]]) -> None:
    """Print a copy-pasteable Markdown table of forecast metrics to stdout."""
    print()
    print("Markdown table (copy into MkDocs):")
    print()
    print("| Model | MAE | RMSE | MAPE (%) |")
    print("|-------|-----|------|----------|")
    for name in MODEL_ORDER:
        if name not in results_dict:
            continue
        metrics = results_dict[name]
        label = MODEL_LABELS.get(name, name)
        print(
            f"| {label} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
            f"{metrics['mape']:.4f} |"
        )
    print()
    print(
        "Note: MAE/RMSE are on normalized consumption (0–1). MAPE can be "
        "unstable when true values are near zero."
    )


def plot_model_comparison(
    predictions: dict[str, pd.DataFrame],
    output_path: Path = FORECAST_COMPARISON_PNG,
    window_steps: int = PLOT_WINDOW_STEPS,
) -> Path:
    """Plot actual vs predicted consumption for each model over a 3-day test window.

    Uses a 2×2 subplot grid (one panel per model) with the last ``window_steps``
    test intervals so lines remain readable at 30-minute resolution.

    Args:
        predictions: Model name → frame with ``timestamp``, ``y_true``, ``y_pred``.
        output_path: PNG destination under ``docs/assets/``.
        window_steps: Number of test steps to display. Defaults to ``144`` (3 days).

    Returns:
        Absolute path to the saved PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes_flat = axes.ravel()

    for ax, name in zip(axes_flat, MODEL_ORDER):
        frame = predictions[name].sort_values("timestamp").reset_index(drop=True)
        window = frame.tail(min(window_steps, len(frame)))

        label = MODEL_LABELS.get(name, name)
        ax.plot(
            window["timestamp"],
            window["y_true"],
            label="Actual",
            color="#1f77b4",
            linewidth=1.5,
        )
        ax.plot(
            window["timestamp"],
            window["y_pred"],
            label="Predicted",
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(f"{label} — last {len(window)} steps (~3 days)")
        ax.set_ylabel("Electricity (normalized)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Forecast comparison — actual vs predicted (normalized consumption)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    _ensure_prophet_available()
    _ensure_xgboost_available()
    _ensure_torch_available()

    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("FORECAST MODEL COMPARISON — PREDICTION COLLECTION")
    print("=" * 60)
    print(f"Artifact: {clean_path}")
    print(f"Target:   {TARGET_COLUMN}")

    df = load_clean_dataset(clean_path)
    train_df, val_df, test_df = time_series_split(df)

    print()
    print("Raw chronological split (naive / Prophet):")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")

    predictions = collect_all_predictions(df)

    for name, frame in predictions.items():
        if len(frame["y_true"]) != len(frame["y_pred"]):
            _fail(f"{name}: y_true and y_pred length mismatch.")

    _print_prediction_summary(predictions)

    results = compute_metrics(predictions)
    print_markdown_table(results)

    plot_path = plot_model_comparison(predictions)
    print()
    print(f"Saved comparison plot: {plot_path}")

    print("=" * 60)
    print("PASS — predictions collected, metrics table and plot generated.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

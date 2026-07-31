"""Score the Phase 3 LSTM regressor on the chronological test set.

Loads the Phase 2 clean CSV, builds sliding-window sequences, splits 70/15/15
in time order, trains ``EnergyLSTM``, and prints MAE, RMSE, and MAPE. Compares
against documented naive, Prophet, and XGBoost baseline floors.

Metrics are computed on **normalized** consumption (0–1 scale from the Kaggle
artifact). No ``StandardScaler`` is applied in this LSTM path, so there is no
inverse transform step. If LSTM-specific scaling is added later, apply
``scaler.inverse_transform`` to both ``y_true`` and ``y_pred`` before scoring.

Run from repository root (use the project ``.venv``)::

    .venv\\Scripts\\activate
    python scripts/evaluate_lstm.py

Or without activating::

    .venv\\Scripts\\python.exe scripts/evaluate_lstm.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.features.build_features import create_sequences  # noqa: E402
from src.models.evaluate_forecast import evaluate_forecast  # noqa: E402
from src.models.lstm_model import EnergyLSTM  # noqa: E402
from src.models.train_forecast_models import (  # noqa: E402
    make_lstm_dataloader,
    predict_lstm,
    train_lstm_model,
)

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
TARGET_COLUMN = "Electricity_Consumed"
SEQ_LENGTH = 24
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_PCT = 0.7
VAL_PCT = 0.15

FEATURE_COLUMNS = [
    TARGET_COLUMN,
    "Temperature",
    "Humidity",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

# Documented baseline floors (reproducible via evaluate_* scripts)
NAIVE_FLOOR_MAE = 0.171150
NAIVE_FLOOR_RMSE = 0.214034
PROPHET_FLOOR_MAE = 0.121071
PROPHET_FLOOR_RMSE = 0.148670
XGBOOST_FLOOR_MAE = 0.125274
XGBOOST_FLOOR_RMSE = 0.153876


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps."""
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/evaluate_lstm.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if TARGET_COLUMN not in df.columns:
        _fail(f"Column '{TARGET_COLUMN}' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("Timestamp", ascending=True).reset_index(drop=True)


def split_sequence_arrays(
    X: np.ndarray,
    y: np.ndarray,
    train_pct: float = TRAIN_PCT,
    val_pct: float = VAL_PCT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split sequence arrays using the same fraction math as ``time_series_split``."""
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


def _ensure_torch_available() -> None:
    """Fail fast with venv guidance when PyTorch is missing from this interpreter."""
    if importlib.util.find_spec("torch") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "PyTorch is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/evaluate_lstm.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/evaluate_lstm.py"
    )


def _compare_to_floor(value: float, floor: float) -> str:
    if value < floor:
        return f"beats floor ({floor:.6f})"
    if value > floor:
        return f"above floor ({floor:.6f})"
    return f"matches floor ({floor:.6f})"


def main() -> None:
    _ensure_torch_available()

    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("LSTM REGRESSOR — TEST SET SCORE")
    print("=" * 60)
    print(f"Artifact: {clean_path}")
    print(f"Target:   {TARGET_COLUMN}")

    df = load_clean_dataset(clean_path)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        _fail(f"Missing feature columns: {missing_features}")

    data = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X, y = create_sequences(data, seq_length=SEQ_LENGTH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_sequence_arrays(X, y)

    print()
    print("Chronological split sizes (after sequence warm-up):")
    print(f"  train: {len(X_train)} sequences")
    print(f"  val:   {len(X_val)} sequences")
    print(f"  test:  {len(X_test)} sequences")
    print(f"  seq_length: {SEQ_LENGTH}")
    print(f"  features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

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
    print(f"Training EnergyLSTM for {EPOCHS} epochs...")
    model = EnergyLSTM(input_size=len(FEATURE_COLUMNS))
    model = train_lstm_model(model, train_loader, val_loader, epochs=EPOCHS)

    y_pred = predict_lstm(model, test_loader)
    y_true = y_test[:, 0]

    if len(y_pred) != len(y_true):
        _fail(
            f"Prediction length mismatch: {len(y_pred)} predictions vs "
            f"{len(y_true)} test targets."
        )

    print()
    print(
        "Note: MAE/RMSE are on normalized scale (0–1); "
        "no inverse transform applied."
    )
    print(
        "      The clean CSV is pre-normalized; this LSTM path does not use "
        "StandardScaler."
    )
    # If LSTM-specific scaling is added later:
    # y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    # y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    metrics = evaluate_forecast(y_true, y_pred)

    print()
    print("LSTM metrics (test set, normalized consumption):")
    print(
        f"  MAE:  {metrics['mae']:.6f}  - "
        f"naive {_compare_to_floor(metrics['mae'], NAIVE_FLOOR_MAE)}; "
        f"Prophet {_compare_to_floor(metrics['mae'], PROPHET_FLOOR_MAE)}; "
        f"XGBoost {_compare_to_floor(metrics['mae'], XGBOOST_FLOOR_MAE)}"
    )
    print(
        f"  RMSE: {metrics['rmse']:.6f}  - "
        f"naive {_compare_to_floor(metrics['rmse'], NAIVE_FLOOR_RMSE)}; "
        f"Prophet {_compare_to_floor(metrics['rmse'], PROPHET_FLOOR_RMSE)}; "
        f"XGBoost {_compare_to_floor(metrics['rmse'], XGBOOST_FLOOR_RMSE)}"
    )
    print(f"  MAPE: {metrics['mape']:.4f} %")
    print()
    print("Baseline floors (reference):")
    print(f"  Naive:   MAE {NAIVE_FLOOR_MAE:.6f}  RMSE {NAIVE_FLOOR_RMSE:.6f}")
    print(f"  Prophet: MAE {PROPHET_FLOOR_MAE:.6f}  RMSE {PROPHET_FLOOR_RMSE:.6f}")
    print(f"  XGBoost: MAE {XGBOOST_FLOOR_MAE:.6f}  RMSE {XGBOOST_FLOOR_RMSE:.6f}")
    print("=" * 60)

    beats_naive = (
        metrics["mae"] < NAIVE_FLOOR_MAE and metrics["rmse"] < NAIVE_FLOOR_RMSE
    )
    beats_prophet = (
        metrics["mae"] < PROPHET_FLOOR_MAE and metrics["rmse"] < PROPHET_FLOOR_RMSE
    )
    if beats_naive and beats_prophet:
        print("PASS - LSTM beats naive and Prophet floors on MAE and RMSE.")
    elif beats_naive:
        print("PASS - LSTM beats the naive floor on MAE and RMSE.")
    else:
        print(
            "NOTE - LSTM did not beat all baseline floors on both MAE and RMSE. "
            "See metrics above."
        )
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

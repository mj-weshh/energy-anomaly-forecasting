"""Score the Phase 3 Prophet statistical baseline on the chronological test set.

Loads the Phase 2 clean CSV, splits 70/15/15 in time order, fits Prophet on
train, forecasts the test window, and prints MAE, RMSE, and MAPE. Compares
against the documented naive seasonal floor (MAE ≈ 0.171, RMSE ≈ 0.214).

Run from repository root (use the project ``.venv`` — Prophet is not on system Python)::

    .venv\\Scripts\\activate
    python scripts/evaluate_prophet.py

Or without activating::

    .venv\\Scripts\\python.exe scripts/evaluate_prophet.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.data.make_forecast_dataset import time_series_split  # noqa: E402
from src.models.evaluate_forecast import evaluate_forecast  # noqa: E402
from src.models.train_forecast_models import train_prophet_model  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
TARGET_COLUMN = "Electricity_Consumed"

# Documented naive seasonal floor from Day 1–2 (reproducible via evaluate_naive_baseline.py)
NAIVE_FLOOR_MAE = 0.171150
NAIVE_FLOOR_RMSE = 0.214034


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps.

    Args:
        path: Absolute path to ``clean_smart_meter_data.csv``.

    Returns:
        DataFrame with parsed ``Timestamp`` column.
    """
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/evaluate_prophet.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if TARGET_COLUMN not in df.columns:
        _fail(f"Column '{TARGET_COLUMN}' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df


def _ensure_prophet_available() -> None:
    """Fail fast with venv guidance when Prophet is missing from this interpreter."""
    if importlib.util.find_spec("prophet") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "Prophet is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Prophet was installed in the project virtual environment. Activate it and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  python scripts/evaluate_prophet.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/evaluate_prophet.py"
    )


def _compare_to_naive(metric_name: str, value: float, floor: float) -> str:
    if value < floor:
        return f"beats naive floor ({floor:.6f})"
    if value > floor:
        return f"above naive floor ({floor:.6f})"
    return f"matches naive floor ({floor:.6f})"


def main() -> None:
    _ensure_prophet_available()

    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("PROPHET STATISTICAL BASELINE — TEST SET SCORE")
    print("=" * 60)
    print(f"Artifact: {clean_path}")
    print(f"Target:   {TARGET_COLUMN}")

    df = load_clean_dataset(clean_path)
    train_df, val_df, test_df = time_series_split(df)

    print()
    print("Chronological split sizes:")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows  (not used for Prophet scoring)")
    print(f"  test:  {len(test_df)} rows")
    print(
        f"  test range: {test_df['Timestamp'].min()} -> {test_df['Timestamp'].max()}"
    )

    print()
    print("Training Prophet on train split and forecasting test window...")
    y_pred = train_prophet_model(train_df, test_df)
    y_test = test_df[TARGET_COLUMN]
    metrics = evaluate_forecast(y_test, y_pred)

    print()
    print("Prophet metrics (test set):")
    print(f"  MAE:  {metrics['mae']:.6f}  - {_compare_to_naive('MAE', metrics['mae'], NAIVE_FLOOR_MAE)}")
    print(f"  RMSE: {metrics['rmse']:.6f}  - {_compare_to_naive('RMSE', metrics['rmse'], NAIVE_FLOOR_RMSE)}")
    print(f"  MAPE: {metrics['mape']:.4f} %")
    print()
    print("Naive seasonal floor (reference):")
    print(f"  MAE:  {NAIVE_FLOOR_MAE:.6f}")
    print(f"  RMSE: {NAIVE_FLOOR_RMSE:.6f}")
    print("=" * 60)

    beats_floor = metrics["mae"] < NAIVE_FLOOR_MAE and metrics["rmse"] < NAIVE_FLOOR_RMSE
    if beats_floor:
        print("PASS - Prophet beats the naive floor on MAE and RMSE.")
    else:
        print(
            "NOTE — Prophet did not beat the naive floor on both MAE and RMSE. "
            "See metrics above."
        )
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

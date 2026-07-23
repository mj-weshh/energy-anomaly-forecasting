"""Score the Phase 3 naive seasonal baseline on the chronological test set.

Loads the Phase 2 clean CSV, splits 70/15/15 in time order, forecasts the
test window with a 48-step (24-hour) seasonal naive model, and prints MAE,
RMSE, and MAPE. The validation split is reported for size only — the naive
baseline needs no hyperparameter tuning.

Run from repository root::

    python scripts/evaluate_naive_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.data.make_forecast_dataset import time_series_split  # noqa: E402
from src.models.evaluate_forecast import evaluate_forecast  # noqa: E402
from src.models.train_forecast_models import naive_seasonal_forecast  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
TARGET_COLUMN = "Electricity_Consumed"
SEASONAL_PERIODS = 48


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
            "  Then re-run:       python scripts/evaluate_naive_baseline.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if TARGET_COLUMN not in df.columns:
        _fail(f"Column '{TARGET_COLUMN}' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("NAIVE SEASONAL BASELINE — TEST SET SCORE")
    print("=" * 60)
    print(f"Artifact: {clean_path}")
    print(f"Target:   {TARGET_COLUMN}")
    print(f"Seasonal periods: {SEASONAL_PERIODS} (24h at 30-min resolution)")

    df = load_clean_dataset(clean_path)
    train_df, val_df, test_df = time_series_split(df)

    print()
    print("Chronological split sizes:")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows  (not used for naive scoring)")
    print(f"  test:  {len(test_df)} rows")
    print(
        f"  test range: {test_df['Timestamp'].min()} → {test_df['Timestamp'].max()}"
    )

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    y_pred = naive_seasonal_forecast(
        y_train, y_test, seasonal_periods=SEASONAL_PERIODS
    )
    metrics = evaluate_forecast(y_test, y_pred)

    print()
    print("Naive seasonal baseline metrics (test set):")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.4f} %")
    print("=" * 60)
    print("PASS — baseline scored. Advanced models must beat these numbers.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Verify supervised lag feature generation for XGBoost prep.

Loads the Phase 2 clean CSV, applies ``create_supervised_lags``, and prints
column names, shape, and a sample of rows so lag alignment can be checked
before model training.

Run from repository root::

    python scripts/verify_xgboost_prep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.features.build_features import create_supervised_lags  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
TARGET_COLUMN = "Electricity_Consumed"
LAG_COLUMNS = [
    f"{TARGET_COLUMN}_lag_1",
    f"{TARGET_COLUMN}_lag_2",
    f"{TARGET_COLUMN}_lag_48",
]


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps."""
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/verify_xgboost_prep.py"
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
    print("XGBOOST PREP — SUPERVISED LAG VERIFICATION")
    print("=" * 60)
    print(f"Artifact: {clean_path}")

    raw_df = load_clean_dataset(clean_path)
    print(f"\nInput shape:  {raw_df.shape}")

    lagged_df = create_supervised_lags(raw_df, target_col=TARGET_COLUMN)
    print(f"Output shape: {lagged_df.shape}")
    print(f"Rows dropped: {len(raw_df) - len(lagged_df)} (expected 48 on continuous series)")

    print("\nColumns:")
    for col in lagged_df.columns:
        print(f"  - {col}")

    print("\nLag columns present:", all(col in lagged_df.columns for col in LAG_COLUMNS))

    print("\nSample rows (target + lags):")
    sample_cols = ["Timestamp", TARGET_COLUMN, *LAG_COLUMNS]
    print(lagged_df[sample_cols].head().to_string(index=False))

    print("=" * 60)
    print("PASS — supervised lag frame ready for inspection.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

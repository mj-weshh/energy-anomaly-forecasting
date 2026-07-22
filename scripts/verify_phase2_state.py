"""Phase 3 Step 0 gate: verify the Phase 2 clean dataset is forecast-ready.

Loads ``data/processed/clean_smart_meter_data.csv`` only — does not regenerate
the artifact or retrain anomaly models.

Checks:
    - File exists (else instruct to run ``generate_clean_data.py``)
    - Exactly 5,000 rows
    - Zero NaNs in ``Electricity_Consumed``
    - Perfect 30-minute timestamp continuity (no gaps, duplicates, or irregular deltas)

Run from repository root::

    python scripts/verify_phase2_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import check_time_continuity, get_project_root  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
EXPECTED_ROWS = 5000
FREQ = "30min"


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
            "  Then re-run:       python scripts/verify_phase2_state.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if "Electricity_Consumed" not in df.columns:
        _fail("Column 'Electricity_Consumed' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def assert_row_count(df: pd.DataFrame) -> None:
    """Require exactly 5,000 rows in the clean artifact."""
    n = len(df)
    if n != EXPECTED_ROWS:
        _fail(f"Expected {EXPECTED_ROWS} rows, found {n}.")


def assert_no_consumption_nans(df: pd.DataFrame) -> None:
    """Require zero NaNs in Electricity_Consumed after interpolation."""
    nan_count = int(df["Electricity_Consumed"].isna().sum())
    if nan_count != 0:
        _fail(f"Electricity_Consumed has {nan_count} NaN(s); expected 0.")


def assert_thirty_minute_continuity(df: pd.DataFrame) -> None:
    """Hard-fail if the series is not a perfect 30-minute grid."""
    ts = df["Timestamp"]
    expected_delta = pd.Timedelta(FREQ)

    duplicate_count = int(ts.duplicated().sum())
    deltas = ts.diff().dropna()
    irregular_count = int((deltas != expected_delta).sum())
    expected_index = pd.date_range(start=ts.min(), end=ts.max(), freq=FREQ)
    missing_count = len(expected_index) - len(set(ts))

    if duplicate_count or irregular_count or missing_count:
        _fail(
            "30-minute continuity broken — "
            f"duplicates={duplicate_count}, "
            f"irregular_intervals={irregular_count}, "
            f"missing_timestamps={missing_count}."
        )


def main() -> None:
    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("PHASE 2 STATE AUDIT (Phase 3 gate)")
    print("=" * 60)
    print(f"Artifact: {clean_path}")

    df = load_clean_dataset(clean_path)
    assert_row_count(df)
    assert_no_consumption_nans(df)
    assert_thirty_minute_continuity(df)

    # Human-readable continuity report (same helper as Phase 1 ingestion).
    check_time_continuity(df, freq=FREQ)

    print("=" * 60)
    print("PASS — Phase 2 clean dataset is forecast-ready.")
    print(f"  Rows:                 {len(df)}")
    print(f"  Electricity_Consumed NaNs: 0")
    print(f"  Continuity:           perfect {FREQ}")
    print(f"  Timestamp range:      {df['Timestamp'].min()} → {df['Timestamp'].max()}")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

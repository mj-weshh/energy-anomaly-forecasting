"""Load and validate the Smart Meter Electricity Consumption Dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_FOLDER_NAME = "Smart Meter Electricity Consumption Dataset"
EXPECTED_FILENAME = "smart_meter_data.csv"


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def find_dataset_csv(root: Path | None = None) -> Path:
    """
    Dynamically locate the smart meter CSV under the project root.

    Search order:
      1. data/raw/*.csv
      2. Smart Meter Electricity Consumption Dataset/*.csv
      3. Any **/smart_meter_data.csv under root
    """
    root = root or get_project_root()

    candidates: list[Path] = []

    raw_dir = root / "data" / "raw"
    if raw_dir.is_dir():
        candidates.extend(sorted(raw_dir.glob("*.csv")))

    dataset_dir = root / DATASET_FOLDER_NAME
    if dataset_dir.is_dir():
        candidates.extend(sorted(dataset_dir.glob("*.csv")))

    fallback = sorted(root.glob(f"**/{EXPECTED_FILENAME}"))
    candidates.extend(fallback)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    if not unique_candidates:
        raise FileNotFoundError(
            f"No CSV found under {root}. Expected one of:\n"
            f"  - {raw_dir / '*.csv'}\n"
            f"  - {dataset_dir / '*.csv'}\n"
            f"  - **/{EXPECTED_FILENAME}"
        )

    return unique_candidates[0]


def load_smart_meter_data(csv_path: Path | str) -> pd.DataFrame:
    """Load CSV, parse Timestamp, and sort chronologically."""
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("Timestamp").reset_index(drop=True)


def print_schema_summary(df: pd.DataFrame) -> None:
    """Print shape, columns, dtypes, and null counts."""
    print("=" * 60)
    print("SCHEMA SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nNull counts:")
    print(df.isna().sum().to_string())
    print("=" * 60)


def check_time_continuity(df: pd.DataFrame, freq: str = "30min") -> None:
    """Verify 30-minute cadence and report gaps, duplicates, and irregular intervals."""
    print("=" * 60)
    print("TIME-SERIES CONTINUITY CHECK")
    print("=" * 60)

    ts = df["Timestamp"]
    expected_delta = pd.Timedelta(freq)

    duplicate_count = int(ts.duplicated().sum())
    deltas = ts.diff().dropna()
    irregular_count = int((deltas != expected_delta).sum())

    expected_index = pd.date_range(start=ts.min(), end=ts.max(), freq=freq)
    observed_set = set(ts)
    missing_timestamps = [t for t in expected_index if t not in observed_set]

    print(f"Frequency:           {freq}")
    print(f"Start timestamp:     {ts.min()}")
    print(f"End timestamp:       {ts.max()}")
    print(f"Observed rows:         {len(df)}")
    print(f"Expected rows:         {len(expected_index)}")
    print(f"Missing timestamps:    {len(missing_timestamps)}")
    print(f"Duplicate timestamps:  {duplicate_count}")
    print(f"Irregular intervals:   {irregular_count}")

    if missing_timestamps:
        print("\nFirst missing timestamps (up to 10):")
        for t in missing_timestamps[:10]:
            print(f"  - {t}")

    if duplicate_count == 0 and irregular_count == 0 and not missing_timestamps:
        print("\nResult: PASS — continuous 30-minute series with no gaps or duplicates.")
    else:
        print("\nResult: REVIEW — continuity issues detected (see counts above).")

    print("=" * 60)


def main() -> None:
    root = get_project_root()
    csv_path = find_dataset_csv(root)
    print(f"Project root: {root}")
    print(f"Loaded CSV:   {csv_path}\n")

    df = load_smart_meter_data(csv_path)
    print_schema_summary(df)
    print()
    check_time_continuity(df)


if __name__ == "__main__":
    main()

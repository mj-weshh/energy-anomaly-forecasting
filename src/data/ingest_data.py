"""Data ingestion and validation for the Smart Meter Electricity Consumption Dataset.

This module is the canonical entry point for loading raw smart meter CSV data
into a validated pandas DataFrame. It handles dynamic file discovery, timestamp
parsing, schema reporting, and 30-minute time-series continuity checks.

Dataset source:
    https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset

Usage:
    Run as a CLI::

        python -m src.data.ingest_data

    Or import in downstream modules::

        from src.data.ingest_data import load_smart_meter_data, find_dataset_csv

Attributes:
    DATASET_FOLDER_NAME: Default folder name for the downloaded Kaggle dataset.
    EXPECTED_FILENAME: Expected CSV filename within the dataset folder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_FOLDER_NAME: str = "Smart Meter Electricity Consumption Dataset"
"""Default folder name created when downloading from Kaggle."""

EXPECTED_FILENAME: str = "smart_meter_data.csv"
"""Expected CSV filename containing 30-minute interval smart meter readings."""


def get_project_root() -> Path:
    """Return the repository root directory.

    Resolved relative to this module's location: ``src/data/ingest_data.py``
    → parent ``src/data/`` → parent ``src/`` → parent repo root.

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[2]


def find_dataset_csv(root: Path | None = None) -> Path:
    """Dynamically locate the smart meter CSV under the project root.

    Searches candidate paths in priority order and returns the first match:

    1. ``data/raw/*.csv``
    2. ``Smart Meter Electricity Consumption Dataset/*.csv``
    3. Any ``**/smart_meter_data.csv`` under ``root``

    Args:
        root: Directory to search. Defaults to :func:`get_project_root`.

    Returns:
        Absolute path to the first matching CSV file.

    Raises:
        FileNotFoundError: If no CSV is found in any candidate location.
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
    """Load and prepare the smart meter CSV for downstream analysis.

    Reads the CSV, parses ``Timestamp`` into datetime objects, and sorts
    rows chronologically.

    Args:
        csv_path: Absolute or relative path to ``smart_meter_data.csv``.

    Returns:
        DataFrame with parsed ``Timestamp`` column, sorted ascending.
        Expected shape after loading: ``(5000, 7)``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the ``Timestamp`` column is missing from the CSV.
        pd.errors.ParserError: If timestamp values cannot be parsed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Timestamp" not in df.columns:
        raise ValueError(
            f"Required column 'Timestamp' not found. Got columns: {list(df.columns)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("Timestamp").reset_index(drop=True)


def print_schema_summary(df: pd.DataFrame) -> None:
    """Print shape, column names, dtypes, and null counts to stdout.

    Args:
        df: Loaded smart meter DataFrame from :func:`load_smart_meter_data`.
    """
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
    """Verify time-series cadence and report gaps, duplicates, and irregular intervals.

    Compares observed timestamps against a complete ``pd.date_range`` at the
    specified frequency. Prints a PASS or REVIEW verdict to stdout.

    Args:
        df: DataFrame with a parsed ``Timestamp`` column.
        freq: Expected pandas frequency string. Defaults to ``"30min"``.

    Raises:
        KeyError: If ``Timestamp`` column is not present in ``df``.
    """
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
    """Run the full ingestion and validation pipeline.

    Discovers the CSV, loads data, prints schema summary, and checks
    time-series continuity. Intended as the CLI entry point::

        python -m src.data.ingest_data
    """
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

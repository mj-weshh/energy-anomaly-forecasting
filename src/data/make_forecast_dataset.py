"""Phase 3 forecasting dataset helpers.

This module provides chronological train / validation / test splitting for
smart-meter time series. Feature and lag builders for forecasting models will
be added in later Phase 3 steps — this file covers the split scaffold only.

Never randomly shuffle rows: that would leak future information into training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def time_series_split(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame chronologically into train, val, and test.

    Rows are ordered by ``Timestamp`` ascending, then sliced by position:
    the first ``train_pct`` fraction for training, the next ``val_pct`` for
    validation, and the remainder for testing (typically ~15% when defaults
    are used).

    This is intentional. Random splits (e.g. ``sklearn.model_selection.train_test_split``)
    would mix future and past intervals and cause data leakage — the model
    would see tomorrow while learning to predict yesterday.

    Args:
        df: Input frame with a ``Timestamp`` column. Other columns are preserved.
        train_pct: Fraction of rows for the training set. Defaults to ``0.7``.
        val_pct: Fraction of rows for the validation set. Defaults to ``0.15``.
            The test fraction is ``1.0 - train_pct - val_pct``.

    Returns:
        A tuple ``(train_df, val_df, test_df)``. Each frame has a fresh
        0-based index (``reset_index(drop=True)``) and the same columns as
        ``df`` after chronological sort.

    Raises:
        KeyError: If ``Timestamp`` is missing from ``df``.
        ValueError: If ``train_pct`` or ``val_pct`` are not in ``(0, 1)``,
            if ``train_pct + val_pct >= 1.0``, or if a resulting split is empty.
    """
    if "Timestamp" not in df.columns:
        raise KeyError("Column 'Timestamp' is required for chronological splitting.")

    if not (0.0 < train_pct < 1.0):
        raise ValueError(f"train_pct must be in (0, 1), got {train_pct}.")
    if not (0.0 < val_pct < 1.0):
        raise ValueError(f"val_pct must be in (0, 1), got {val_pct}.")
    if train_pct + val_pct >= 1.0:
        raise ValueError(
            f"train_pct + val_pct must be < 1.0 so a test set remains; "
            f"got {train_pct + val_pct}."
        )

    ordered = df.sort_values("Timestamp", ascending=True)
    n = len(ordered)
    if n == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    val_df = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test_df = ordered.iloc[val_end:].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "One or more splits are empty after slicing; "
            f"n={n}, train_end={train_end}, val_end={val_end}."
        )

    return train_df, val_df, test_df


def _print_split_summary(name: str, split_df: pd.DataFrame) -> None:
    """Print row count and timestamp bounds for one split."""
    start = split_df["Timestamp"].min()
    end = split_df["Timestamp"].max()
    print(f"{name}:")
    print(f"  rows:  {len(split_df)}")
    print(f"  start: {start}")
    print(f"  end:   {end}")


def _main() -> None:
    """Load the Phase 2 clean CSV, split chronologically, verify boundaries."""
    repo_root = Path(__file__).resolve().parents[2]
    clean_path = repo_root / "data" / "processed" / "clean_smart_meter_data.csv"

    print("=" * 60)
    print("CHRONOLOGICAL SPLIT BOUNDARY CHECK")
    print("=" * 60)
    print(f"Artifact: {clean_path}")

    if not clean_path.is_file():
        print(
            f"FAIL: Clean dataset not found at {clean_path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python -m src.data.make_forecast_dataset",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(clean_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")

    train_df, val_df, test_df = time_series_split(df)

    print()
    _print_split_summary("train", train_df)
    _print_split_summary("val", val_df)
    _print_split_summary("test", test_df)

    train_end = train_df["Timestamp"].max()
    val_start = val_df["Timestamp"].min()
    val_end = val_df["Timestamp"].max()
    test_start = test_df["Timestamp"].min()
    step = pd.Timedelta("30min")

    print()
    print("Adjacency checks:")
    ok = True

    train_val_ok = train_end < val_start and val_start == train_end + step
    if train_val_ok:
        print("  train → val: PASS (train end precedes val start by 30 min)")
    else:
        ok = False
        print("  train → val: FAIL")
    print(f"    train end: {train_end}")
    print(f"    val start: {val_start}")

    val_test_ok = val_end < test_start and test_start == val_end + step
    if val_test_ok:
        print("  val → test: PASS (val end precedes test start by 30 min)")
    else:
        ok = False
        print("  val → test: FAIL")
    print(f"    val end:    {val_end}")
    print(f"    test start: {test_start}")

    print("=" * 60)
    if ok:
        print("PASS — chronological split boundaries OK")
        print("=" * 60)
        sys.exit(0)

    print("FAIL — chronological split boundaries are invalid", file=sys.stderr)
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    _main()

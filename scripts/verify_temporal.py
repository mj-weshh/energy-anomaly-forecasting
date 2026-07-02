"""Verify temporal feature extraction on the real smart meter dataset.

Loads data through the canonical ingestion module, applies
``add_temporal_features``, prints the head of the new columns, and runs
basic sanity checks.

Run from repository root::

    python scripts/verify_temporal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.features.build_features import add_temporal_features  # noqa: E402

TEMPORAL_COLUMNS = ["hour", "day_of_week", "month", "is_weekend"]


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = add_temporal_features(load_smart_meter_data(csv_path))

    print(f"Loaded: {csv_path}")
    print(f"Shape after feature extraction: {df.shape}\n")
    print(df[["Timestamp", *TEMPORAL_COLUMNS]].head().to_string(index=False))
    print()

    missing = [col for col in TEMPORAL_COLUMNS if col not in df.columns]
    assert not missing, f"Missing temporal columns: {missing}"
    assert df["hour"].between(0, 23).all(), "hour outside 0-23 range"
    assert df["day_of_week"].between(0, 6).all(), "day_of_week outside 0-6 range"
    assert df["month"].between(1, 12).all(), "month outside 1-12 range"
    assert set(df["is_weekend"].unique()) <= {0, 1}, "is_weekend not binary"

    print("PASS — all temporal columns present with valid value ranges.")


if __name__ == "__main__":
    main()

"""Verify feature engineering on the real smart meter dataset.

Loads data through the canonical ingestion module, applies
``add_temporal_features`` and ``add_rolling_metrics``, prints sample rows,
and runs basic sanity checks on all engineered columns.

Run from repository root::

    python scripts/verify_features.py
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
from src.features.build_features import (  # noqa: E402
    add_rolling_metrics,
    add_temporal_features,
)

TEMPORAL_COLUMNS = ["hour", "day_of_week", "month", "is_weekend"]
ROLLING_COLUMNS = [
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_mean_24h",
    "rolling_std_24h",
]
WINDOW_3H = 6
WINDOW_24H = 48


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = add_rolling_metrics(add_temporal_features(load_smart_meter_data(csv_path)))

    print(f"Loaded: {csv_path}")
    print(f"Shape after feature engineering: {df.shape}\n")

    print("Temporal features (head):")
    print(df[["Timestamp", *TEMPORAL_COLUMNS]].head().to_string(index=False))
    print()

    print("Rolling metrics (tail — windows fully filled):")
    print(df[["Timestamp", *ROLLING_COLUMNS]].tail().to_string(index=False))
    print()

    missing = [
        col for col in TEMPORAL_COLUMNS + ROLLING_COLUMNS if col not in df.columns
    ]
    assert not missing, f"Missing feature columns: {missing}"

    assert df["hour"].between(0, 23).all(), "hour outside 0-23 range"
    assert df["day_of_week"].between(0, 6).all(), "day_of_week outside 0-6 range"
    assert df["month"].between(1, 12).all(), "month outside 1-12 range"
    assert set(df["is_weekend"].unique()) <= {0, 1}, "is_weekend not binary"

    assert not df[ROLLING_COLUMNS].tail().isna().any().any(), (
        "rolling metrics contain NaN after window fill"
    )
    assert df["rolling_mean_3h"].head(WINDOW_3H - 1).isna().all(), (
        "rolling_mean_3h warm-up rows should be NaN"
    )
    assert df["rolling_mean_24h"].head(WINDOW_24H - 1).isna().all(), (
        "rolling_mean_24h warm-up rows should be NaN"
    )

    print("PASS — temporal and rolling feature columns present with valid values.")


if __name__ == "__main__":
    main()

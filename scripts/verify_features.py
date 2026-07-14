"""Verify feature engineering on the real smart meter dataset.

Loads data through the canonical ingestion module, applies
``add_temporal_features`` and ``add_rolling_metrics``, prints sample rows,
and runs basic sanity checks on all engineered columns.

Run from repository root::

    python scripts/verify_features.py
    python scripts/verify_features.py --enhanced
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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
    build_enhanced_anomaly_features,
)

TEMPORAL_COLUMNS = ["hour", "day_of_week", "month", "is_weekend"]
ROLLING_COLUMNS = [
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_mean_24h",
    "rolling_std_24h",
]
CYCLICAL_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
DERIVATIVE_COLUMNS = ["consumption_diff", "consumption_residual_24h"]
WINDOW_3H = 6
WINDOW_24H = 48


def verify_legacy(df) -> None:
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


def verify_enhanced(df) -> None:
    verify_legacy(df)

    missing = [
        col for col in CYCLICAL_COLUMNS + DERIVATIVE_COLUMNS if col not in df.columns
    ]
    assert not missing, f"Missing enhanced columns: {missing}"
    assert df.shape == (5000, 21), f"Expected (5000, 21), got {df.shape}"

    for col in CYCLICAL_COLUMNS:
        assert df[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"

    assert pd.isna(df["consumption_diff"].iloc[0]), (
        "consumption_diff first row should be NaN"
    )
    assert not df["consumption_residual_24h"].tail().isna().any(), (
        "consumption_residual_24h NaN at tail after rolling fill"
    )

    print("PASS — enhanced cyclical and derivative columns present with valid values.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify feature engineering.")
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Validate build_enhanced_anomaly_features (21 columns).",
    )
    args = parser.parse_args()

    csv_path = find_dataset_csv(get_project_root())
    raw = load_smart_meter_data(csv_path)

    if args.enhanced:
        df = build_enhanced_anomaly_features(raw)
        verify_enhanced(df)
    else:
        df = add_rolling_metrics(add_temporal_features(raw))
        verify_legacy(df)

    print(f"\nLoaded: {csv_path}")


if __name__ == "__main__":
    main()

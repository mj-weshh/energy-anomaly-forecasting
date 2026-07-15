"""Generate the Phase 3 clean smart meter dataset.

Runs the full clean-data pipeline (load → features → anomaly detection
→ time interpolation) and writes the artifact under ``data/processed/``.

Run from repository root::

    python scripts/generate_clean_data.py
    python scripts/generate_clean_data.py --profile legacy_threshold
    python scripts/generate_clean_data.py --profile enhanced
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import find_dataset_csv, get_project_root  # noqa: E402
from src.pipelines.clean_dataset import DEFAULT_OUTPUTS, CleanProfile, generate_clean_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean smart meter CSV.")
    parser.add_argument(
        "--profile",
        choices=["legacy", "legacy_threshold", "enhanced"],
        default="legacy",
        help="Clean-data detection profile (default: legacy, unchanged).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: data/processed/<profile filename>).",
    )
    args = parser.parse_args()
    profile: CleanProfile = args.profile

    root = get_project_root()
    input_path = find_dataset_csv(root)
    output_path = args.output or (root / "data" / "processed" / DEFAULT_OUTPUTS[profile])

    written = generate_clean_dataset(str(input_path), str(output_path), profile=profile)
    df = pd.read_csv(written)

    print(f"Profile: {profile}")
    print(f"Loaded: {input_path}")
    print(f"Wrote:  {written}")
    print(f"Shape:  {df.shape}")
    print(f"Electricity_Consumed NaNs: {df['Electricity_Consumed'].isna().sum()}")


if __name__ == "__main__":
    main()

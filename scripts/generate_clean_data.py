"""Generate the Phase 3 clean smart meter dataset.

Runs the full clean-data pipeline (load → features → Isolation Forest
detection → time interpolation) and writes the artifact to
``data/processed/clean_smart_meter_data.csv``.

Run from repository root::

    python scripts/generate_clean_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clean_data import generate_clean_dataset  # noqa: E402
from src.data.ingest_data import find_dataset_csv, get_project_root  # noqa: E402

OUTPUT_REL = Path("data/processed/clean_smart_meter_data.csv")


def main() -> None:
    root = get_project_root()
    input_path = find_dataset_csv(root)
    output_path = root / OUTPUT_REL

    written = generate_clean_dataset(str(input_path), str(output_path))
    df = pd.read_csv(written)

    print(f"Loaded: {input_path}")
    print(f"Wrote:  {written}")
    print(f"Shape:  {df.shape}")
    print(f"Electricity_Consumed NaNs: {df['Electricity_Consumed'].isna().sum()}")


if __name__ == "__main__":
    main()

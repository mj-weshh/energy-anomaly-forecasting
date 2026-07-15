"""Compare clean-data artifacts produced by different detection profiles.

Run from repository root after generating artifacts::

    python scripts/generate_clean_data.py --profile legacy
    python scripts/generate_clean_data.py --profile legacy_threshold
    python scripts/generate_clean_data.py --profile enhanced
    python scripts/compare_clean_artifacts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data  # noqa: E402
from src.features.build_features import build_all_features  # noqa: E402
from src.pipelines.clean_dataset import DEFAULT_OUTPUTS  # noqa: E402


def _aligned_consumption(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Timestamp").reset_index(drop=True)


def _imputed_mask(baseline: pd.DataFrame, artifact: pd.DataFrame) -> np.ndarray:
    base = _aligned_consumption(baseline)
    art = _aligned_consumption(artifact)
    return (base["Electricity_Consumed"].to_numpy() != art["Electricity_Consumed"].to_numpy())


def _compare_pair(
    name_a: str,
    df_a: pd.DataFrame,
    name_b: str,
    df_b: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | int]:
    imputed_a = _imputed_mask(baseline, df_a)
    imputed_b = _imputed_mask(baseline, df_b)
    disagree = imputed_a ^ imputed_b

    art_a = _aligned_consumption(df_a)
    art_b = _aligned_consumption(df_b)
    both_imputed = imputed_a & imputed_b
    diffs = np.abs(
        art_a.loc[both_imputed, "Electricity_Consumed"].to_numpy()
        - art_b.loc[both_imputed, "Electricity_Consumed"].to_numpy()
    )

    hour_disagree = (
        art_a.loc[disagree, "hour"].value_counts().sort_index()
        if "hour" in art_a.columns and disagree.any()
        else pd.Series(dtype=int)
    )

    return {
        "pair": f"{name_a} vs {name_b}",
        "imputed_rows_a": int(imputed_a.sum()),
        "imputed_rows_b": int(imputed_b.sum()),
        "disagree_rows": int(disagree.sum()),
        "jaccard_imputed": float(
            (imputed_a & imputed_b).sum() / max((imputed_a | imputed_b).sum(), 1)
        ),
        "max_abs_diff_imputed": float(diffs.max()) if len(diffs) else 0.0,
        "mean_abs_diff_imputed": float(diffs.mean()) if len(diffs) else 0.0,
        "top_disagree_hour": int(hour_disagree.idxmax()) if not hour_disagree.empty else -1,
        "top_disagree_hour_count": int(hour_disagree.max()) if not hour_disagree.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clean-data CSV artifacts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths only; do not read CSVs.",
    )
    args = parser.parse_args()

    root = get_project_root()
    processed = root / "data" / "processed"
    paths = {name: processed / filename for name, filename in DEFAULT_OUTPUTS.items()}

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        print("Missing artifacts (run generate_clean_data.py for each profile):")
        for name in missing:
            print(f"  - {name}: {paths[name]}")
        sys.exit(1)

    if args.dry_run:
        print("Dry run OK — all artifact paths exist.")
        return

    raw = load_smart_meter_data(find_dataset_csv(root))
    baseline = build_all_features(raw)
    artifacts = {name: pd.read_csv(path) for name, path in paths.items()}

    print("Clean artifact comparison (baseline = raw feature-engineered consumption)\n")
    pairs = [
        _compare_pair("legacy", artifacts["legacy"], "legacy_threshold", artifacts["legacy_threshold"], baseline),
        _compare_pair("legacy", artifacts["legacy"], "enhanced", artifacts["enhanced"], baseline),
        _compare_pair(
            "legacy_threshold",
            artifacts["legacy_threshold"],
            "enhanced",
            artifacts["enhanced"],
            baseline,
        ),
    ]

    for row in pairs:
        print(f"{row['pair']}:")
        print(f"  imputed rows: {row['imputed_rows_a']} / {row['imputed_rows_b']}")
        print(f"  disagree imputation flags: {row['disagree_rows']}")
        print(f"  Jaccard (imputed overlap): {row['jaccard_imputed']:.3f}")
        print(f"  mean |diff| on shared imputed: {row['mean_abs_diff_imputed']:.4f}")
        print(
            f"  hour with most disagreements: {row['top_disagree_hour']:02d} "
            f"({row['top_disagree_hour_count']} rows)"
        )
        print()


if __name__ == "__main__":
    main()

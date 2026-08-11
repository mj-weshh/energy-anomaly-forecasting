"""Train Phase 3 XGBoost and export gain-based feature importance.

Uses the same clean CSV, lag prep, chronological split, feature list, and
``train_xgboost_model`` path as ``evaluate_xgboost.py``. Writes a horizontal
bar chart to ``docs/assets/xgboost_feature_importance.png`` for the research
write-up.

Run from repository root (use the project ``.venv``)::

    python scripts/export_xgboost_feature_importance.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.data.make_forecast_dataset import time_series_split  # noqa: E402
from src.features.build_features import create_supervised_lags  # noqa: E402
from src.models.train_forecast_models import train_xgboost_model  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
OUTPUT_RELATIVE_PATH = Path("docs") / "assets" / "xgboost_feature_importance.png"
TARGET_COLUMN = "Electricity_Consumed"

FEATURE_COLUMNS = [
    f"{TARGET_COLUMN}_lag_1",
    f"{TARGET_COLUMN}_lag_2",
    f"{TARGET_COLUMN}_lag_48",
    "Temperature",
    "Humidity",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _ensure_xgboost_available() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        return
    _fail(
        "XGBoost is not installed for this Python interpreter.\n"
        "  Activate .venv and: pip install -r requirements.txt"
    )


def load_clean_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py"
        )
    df = pd.read_csv(path)
    if "Timestamp" not in df.columns or TARGET_COLUMN not in df.columns:
        _fail("Clean CSV missing Timestamp or Electricity_Consumed.")
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    _ensure_xgboost_available()

    root = get_project_root()
    clean_path = root / CLEAN_RELATIVE_PATH
    out_path = root / OUTPUT_RELATIVE_PATH

    print("=" * 60)
    print("XGBOOST FEATURE IMPORTANCE (gain)")
    print("=" * 60)
    print(f"Artifact: {clean_path}")

    df = load_clean_dataset(clean_path)
    tabular_df = create_supervised_lags(df, target_col=TARGET_COLUMN)
    missing = [c for c in FEATURE_COLUMNS if c not in tabular_df.columns]
    if missing:
        _fail(f"Missing feature columns: {missing}")

    train_df, val_df, _test_df = time_series_split(tabular_df)
    model = train_xgboost_model(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        val_df[FEATURE_COLUMNS],
        val_df[TARGET_COLUMN],
    )

    # Gain importance from the booster (sum of improvement), aligned to feature names.
    booster = model.get_booster()
    score_map = booster.get_score(importance_type="gain")
    # sklearn feature names may be f0..fN if not set; use FEATURE_COLUMNS order.
    gains = []
    for i, name in enumerate(FEATURE_COLUMNS):
        key = name if name in score_map else f"f{i}"
        gains.append(float(score_map.get(key, 0.0)))

    order = np.argsort(gains)
    names_sorted = [FEATURE_COLUMNS[i] for i in order]
    gains_sorted = [gains[i] for i in order]

    print()
    print("Feature importance (gain), ascending:")
    for name, gain in zip(names_sorted, gains_sorted):
        print(f"  {name:32s}  {gain:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted, gains_sorted, color="steelblue")
    ax.set_xlabel("Gain (XGBoost)")
    ax.set_title("XGBoost feature importance — forecast model (default train)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Saved: {out_path.resolve()}")
    print("PASS — feature importance chart written.")


if __name__ == "__main__":
    main()

"""Score the Phase 3 XGBoost regressor on the chronological test set.

Loads the Phase 2 clean CSV, builds supervised lag features, splits 70/15/15
in time order, trains XGBoost on tabular predictors, and prints MAE, RMSE,
and MAPE. Compares against documented naive and Prophet baseline floors.

Run from repository root (use the project ``.venv``)::

    .venv\\Scripts\\activate
    python scripts/evaluate_xgboost.py

Or without activating::

    .venv\\Scripts\\python.exe scripts/evaluate_xgboost.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.data.make_forecast_dataset import time_series_split  # noqa: E402
from src.features.build_features import create_supervised_lags  # noqa: E402
from src.models.evaluate_forecast import evaluate_forecast  # noqa: E402
from src.models.train_forecast_models import train_xgboost_model  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
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

# Documented baseline floors (reproducible via evaluate_* scripts)
NAIVE_FLOOR_MAE = 0.171150
NAIVE_FLOOR_RMSE = 0.214034
PROPHET_FLOOR_MAE = 0.121071
PROPHET_FLOOR_RMSE = 0.148670


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps."""
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/evaluate_xgboost.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")
    if TARGET_COLUMN not in df.columns:
        _fail(f"Column '{TARGET_COLUMN}' missing from clean dataset.")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df


def _ensure_xgboost_available() -> None:
    """Fail fast with venv guidance when XGBoost is missing from this interpreter."""
    if importlib.util.find_spec("xgboost") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "XGBoost is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/evaluate_xgboost.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/evaluate_xgboost.py"
    )


def _compare_to_floor(value: float, floor: float) -> str:
    if value < floor:
        return f"beats floor ({floor:.6f})"
    if value > floor:
        return f"above floor ({floor:.6f})"
    return f"matches floor ({floor:.6f})"


def main() -> None:
    _ensure_xgboost_available()

    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("XGBOOST REGRESSOR — TEST SET SCORE")
    print("=" * 60)
    print(f"Artifact: {clean_path}")
    print(f"Target:   {TARGET_COLUMN}")

    df = load_clean_dataset(clean_path)
    tabular_df = create_supervised_lags(df, target_col=TARGET_COLUMN)

    missing_features = [col for col in FEATURE_COLUMNS if col not in tabular_df.columns]
    if missing_features:
        _fail(f"Missing feature columns: {missing_features}")

    train_df, val_df, test_df = time_series_split(tabular_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print()
    print("Chronological split sizes (after lag warm-up):")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")
    print(
        f"  test range: {test_df['Timestamp'].min()} -> {test_df['Timestamp'].max()}"
    )
    print(f"  features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

    print()
    print("Training XGBoost on train split (val used for eval_set)...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    metrics = evaluate_forecast(y_test, y_pred)

    print()
    print("XGBoost metrics (test set):")
    print(
        f"  MAE:  {metrics['mae']:.6f}  - "
        f"naive {_compare_to_floor(metrics['mae'], NAIVE_FLOOR_MAE)}; "
        f"Prophet {_compare_to_floor(metrics['mae'], PROPHET_FLOOR_MAE)}"
    )
    print(
        f"  RMSE: {metrics['rmse']:.6f}  - "
        f"naive {_compare_to_floor(metrics['rmse'], NAIVE_FLOOR_RMSE)}; "
        f"Prophet {_compare_to_floor(metrics['rmse'], PROPHET_FLOOR_RMSE)}"
    )
    print(f"  MAPE: {metrics['mape']:.4f} %")
    print()
    print("Baseline floors (reference):")
    print(f"  Naive:  MAE {NAIVE_FLOOR_MAE:.6f}  RMSE {NAIVE_FLOOR_RMSE:.6f}")
    print(f"  Prophet: MAE {PROPHET_FLOOR_MAE:.6f}  RMSE {PROPHET_FLOOR_RMSE:.6f}")
    print("=" * 60)

    beats_naive = (
        metrics["mae"] < NAIVE_FLOOR_MAE and metrics["rmse"] < NAIVE_FLOOR_RMSE
    )
    beats_prophet = (
        metrics["mae"] < PROPHET_FLOOR_MAE and metrics["rmse"] < PROPHET_FLOOR_RMSE
    )
    if beats_naive and beats_prophet:
        print("PASS - XGBoost beats naive and Prophet floors on MAE and RMSE.")
    elif beats_naive:
        print("PASS - XGBoost beats the naive floor on MAE and RMSE.")
    else:
        print(
            "NOTE - XGBoost did not beat all baseline floors on both MAE and RMSE. "
            "See metrics above."
        )
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Verify 3D LSTM sequence generation and PyTorch tensor conversion.

Loads a small slice of the Phase 2 clean CSV, builds sliding-window sequences
via ``create_sequences``, and converts the feature tensor to ``torch.float32``.

Run from repository root (use the project ``.venv``)::

    .venv\\Scripts\\activate
    python scripts/verify_lstm_prep.py

Or without activating::

    .venv\\Scripts\\python.exe scripts/verify_lstm_prep.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import get_project_root  # noqa: E402
from src.features.build_features import create_sequences  # noqa: E402

CLEAN_RELATIVE_PATH = Path("data") / "processed" / "clean_smart_meter_data.csv"
SEQ_LENGTH = 24
SLICE_ROWS = 200

FEATURE_COLUMNS = [
    "Electricity_Consumed",
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


def _ensure_torch_available() -> None:
    """Fail fast with venv guidance when PyTorch is missing from this interpreter."""
    if importlib.util.find_spec("torch") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _fail(
        "PyTorch is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "Install in the project virtual environment and re-run:\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/verify_lstm_prep.py\n\n"
        "Or run directly:\n"
        f"  {venv_python} scripts/verify_lstm_prep.py"
    )


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Load the Phase 2 clean CSV and parse timestamps."""
    if not path.is_file():
        _fail(
            f"Clean dataset not found at {path}.\n"
            "  Generate it with:  python scripts/generate_clean_data.py\n"
            "  Then re-run:       python scripts/verify_lstm_prep.py"
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        _fail("Column 'Timestamp' missing from clean dataset.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        _fail(f"Missing feature columns: {missing}")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("Timestamp").reset_index(drop=True)


def main() -> None:
    _ensure_torch_available()
    import torch

    clean_path = get_project_root() / CLEAN_RELATIVE_PATH
    print("=" * 60)
    print("LSTM PREP — 3D SEQUENCE AND TENSOR VERIFICATION")
    print("=" * 60)
    print(f"Artifact:   {clean_path}")
    print(f"Seq length: {SEQ_LENGTH} steps (12 h at 30-min resolution)")
    print(f"Slice rows: {SLICE_ROWS}")

    df = load_clean_dataset(clean_path)
    slice_df = df.iloc[:SLICE_ROWS]
    data = slice_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    print(f"\nInput 2D shape: {data.shape}")

    X, y = create_sequences(data, seq_length=SEQ_LENGTH)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    num_features = len(FEATURE_COLUMNS)
    expected_samples = SLICE_ROWS - SEQ_LENGTH
    expected_shape = (expected_samples, SEQ_LENGTH, num_features)

    print(f"NumPy X shape:  {X.shape}")
    print(f"NumPy y shape:  {y.shape}")
    print(f"X tensor shape: {tuple(X_tensor.shape)}")

    if X_tensor.shape != expected_shape:
        _fail(
            f"Expected X tensor shape {expected_shape}; got {tuple(X_tensor.shape)}."
        )

    print(f"\nSample count check: {X_tensor.shape[0]} = {SLICE_ROWS} - {SEQ_LENGTH}")
    print("=" * 60)
    print("PASS — 3D sequence tensor ready for LSTM input.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Anomaly masking and interpolation for Phase 3 forecasting prep.

Phase 3 time-series models (ARIMA, LSTM) require continuous 30-minute
intervals. Anomalous consumption readings must be imputed, not dropped.
This module masks model-flagged intervals on ``Electricity_Consumed`` and
fills gaps with time-aware interpolation while preserving row count.

Typical upstream flow: feature engineering → anomaly predictions →
:func:`interpolate_anomalies`. Orchestration lives in
``src.pipelines.clean_dataset``. See docs/anomaly-detection.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.feature_matrix import prepare_feature_matrix

# Backward-compatible re-export; prefer ``src.pipelines.clean_dataset``.
from src.pipelines.clean_dataset import generate_clean_dataset  # noqa: F401


def interpolate_anomalies(
    df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Mask predicted anomalies and time-interpolate ``Electricity_Consumed``.

    Where ``predictions == 1`` (Abnormal), sets ``Electricity_Consumed`` to
    ``NaN`` then fills gaps via ``interpolate(method="time")`` on a temporary
    DatetimeIndex. Row count and all other columns are preserved.

    Predictions may be shorter than ``df`` when they come from
    ``detect_anomalies`` (4953 scored rows after rolling warm-up). In that
    case, rows are aligned via :func:`prepare_feature_matrix` index; warm-up
    rows keep their original consumption values.

    Args:
        df: DataFrame with ``Timestamp`` and ``Electricity_Consumed`` columns.
            Should be feature-engineered when using model predictions.
        predictions: Binary array (``0`` = Normal, ``1`` = Abnormal). Length
            must equal ``len(df)`` or ``len(prepare_feature_matrix(df))``.

    Returns:
        Copy of ``df`` with anomalous consumption imputed. Same shape and
        columns as input.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If ``predictions`` length does not match ``df`` or the
            evaluable row count, or contains values other than 0/1.
    """
    for column in ("Timestamp", "Electricity_Consumed"):
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' not found in DataFrame.")

    pred_arr = np.asarray(predictions, dtype=int)
    if pred_arr.size == 0:
        raise ValueError("predictions must be non-empty.")
    if not set(np.unique(pred_arr)).issubset({0, 1}):
        raise ValueError("predictions must contain only 0 and 1.")

    result = df.copy()
    result = result.sort_values("Timestamp").reset_index(drop=True)

    if len(pred_arr) == len(result):
        anomaly_mask = pred_arr == 1
    else:
        eval_index = prepare_feature_matrix(result).index
        if len(pred_arr) != len(eval_index):
            raise ValueError(
                f"predictions length ({len(pred_arr)}) must match df length "
                f"({len(result)}) or evaluable row count ({len(eval_index)})."
            )
        mask_series = pd.Series(False, index=result.index)
        mask_series.loc[eval_index] = pred_arr == 1
        anomaly_mask = mask_series.to_numpy()

    if not anomaly_mask.any():
        return result

    result.loc[anomaly_mask, "Electricity_Consumed"] = np.nan

    consumption = (
        result.set_index("Timestamp")["Electricity_Consumed"]
        .interpolate(method="time")
        .to_numpy()
    )
    result["Electricity_Consumed"] = consumption

    return result

"""Forecasting evaluation metrics for Phase 3.

Standardized MAE, RMSE, and MAPE so every forecasting model (naive baseline,
Prophet/ARIMA, XGBoost, LSTM) is scored the same way on held-out windows.

Separate from :mod:`src.models.evaluate_models`, which scores Phase 2 anomaly
detection (precision / recall / F1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _as_1d_float_arrays(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1-D float arrays of equal length without mutating callers.

    Args:
        y_true: Ground-truth target values (array-like).
        y_pred: Predicted target values (array-like).

    Returns:
        Tuple ``(y_true_arr, y_pred_arr)`` as ``float64`` vectors.

    Raises:
        ValueError: If lengths differ or either input is empty.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: "
            f"{y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}."
        )
    return y_true_arr, y_pred_arr


def mean_absolute_error_forecast(y_true: Any, y_pred: Any) -> float:
    """Compute Mean Absolute Error (MAE) for a forecast.

    MAE is the average absolute miss — easy to explain to management.

    Args:
        y_true: Ground-truth target values (array-like).
        y_pred: Predicted target values (array-like).

    Returns:
        MAE as a Python ``float``.
    """
    y_true_arr, y_pred_arr = _as_1d_float_arrays(y_true, y_pred)
    return float(mean_absolute_error(y_true_arr, y_pred_arr))


def root_mean_squared_error_forecast(y_true: Any, y_pred: Any) -> float:
    """Compute Root Mean Squared Error (RMSE) for a forecast.

    RMSE penalizes large misses more heavily than MAE.

    Args:
        y_true: Ground-truth target values (array-like).
        y_pred: Predicted target values (array-like).

    Returns:
        RMSE as a Python ``float``.
    """
    y_true_arr, y_pred_arr = _as_1d_float_arrays(y_true, y_pred)
    return float(mean_squared_error(y_true_arr, y_pred_arr, squared=False))


def mean_absolute_percentage_error_forecast(
    y_true: Any,
    y_pred: Any,
    epsilon: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) with a zero-safe denominator.

    Uses ``max(|y_true|, epsilon)`` in the denominator so near-zero true values
    (common on this normalized 0–1 consumption scale) do not cause division by
    zero or explode the metric. The result is expressed as a **percentage**
    (e.g. ``12.5`` means 12.5%), not a fraction.

    Args:
        y_true: Ground-truth target values (array-like).
        y_pred: Predicted target values (array-like).
        epsilon: Floor applied to ``|y_true|`` before division. Defaults to
            ``1e-8``.

    Returns:
        MAPE in percent as a Python ``float``.

    Raises:
        ValueError: If ``epsilon`` is not positive.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    y_true_arr, y_pred_arr = _as_1d_float_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true_arr), epsilon)
    mape = np.mean(np.abs(y_true_arr - y_pred_arr) / denom) * 100.0
    return float(mape)


def evaluate_forecast(
    y_true: Any,
    y_pred: Any,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a forecast in one call.

    Args:
        y_true: Ground-truth target values (array-like).
        y_pred: Predicted target values (array-like).
        epsilon: Floor for MAPE denominators. Defaults to ``1e-8``.

    Returns:
        Dictionary with keys ``mae``, ``rmse``, and ``mape`` (MAPE in percent).
    """
    return {
        "mae": mean_absolute_error_forecast(y_true, y_pred),
        "rmse": root_mean_squared_error_forecast(y_true, y_pred),
        "mape": mean_absolute_percentage_error_forecast(
            y_true, y_pred, epsilon=epsilon
        ),
    }

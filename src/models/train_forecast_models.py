"""Phase 3 forecasting model trainers.

Starts with a seasonal naive baseline. Advanced models (Prophet/ARIMA, XGBoost,
LSTM) land in later Phase 3 weeks — they must beat this floor on the same
held-out test window to be considered useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_seasonal_forecast(
    train_series: pd.Series | np.ndarray | list[float],
    test_series: pd.Series | np.ndarray | list[float],
    seasonal_periods: int = 48,
) -> np.ndarray:
    """Forecast the test window by seasonal persistence (same time yesterday).

    Energy load has strong daily seasonality. At 30-minute resolution, one day
    is ``48`` steps, so the naive guess for any interval is the observed value
    exactly ``seasonal_periods`` steps earlier.

    Lag lookup uses **observed** history only: the chronological concatenation
    of ``train_series`` and true ``test_series``. Predictions are not fed back
    into the lag (not recursive). That matches the classic seasonal-naive
    baseline used when scoring a full test set at once.

    Args:
        train_series: Chronological training targets (array-like).
        test_series: Chronological test targets (array-like). Length defines
            how many forecasts to produce; true values inside the test window
            are used only as lag history for later test steps.
        seasonal_periods: Seasonal cycle length in steps. Defaults to ``48``
            (24 hours at 30-minute intervals).

    Returns:
        ``numpy.ndarray`` of shape ``(len(test_series),)`` with float forecasts.

    Raises:
        ValueError: If ``seasonal_periods < 1``, either series is empty, or
            ``len(train_series) < seasonal_periods``.
    """
    if seasonal_periods < 1:
        raise ValueError(
            f"seasonal_periods must be >= 1, got {seasonal_periods}."
        )

    train = np.asarray(train_series, dtype=np.float64).reshape(-1)
    test = np.asarray(test_series, dtype=np.float64).reshape(-1)

    if train.size == 0:
        raise ValueError("train_series must be non-empty.")
    if test.size == 0:
        raise ValueError("test_series must be non-empty.")
    if train.size < seasonal_periods:
        raise ValueError(
            f"train_series length ({train.size}) must be >= seasonal_periods "
            f"({seasonal_periods})."
        )

    # Full observed timeline for lag lookup (train then true test).
    y = np.concatenate([train, test])
    n_train = train.size
    preds = np.empty(test.size, dtype=np.float64)

    for i in range(test.size):
        t = n_train + i
        preds[i] = y[t - seasonal_periods]

    return preds


def _prophet_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``Timestamp`` / ``Electricity_Consumed`` and rename to Prophet ``ds`` / ``y``."""
    if "Timestamp" not in df.columns:
        raise KeyError("Column 'Timestamp' is required for Prophet formatting.")
    if "Electricity_Consumed" not in df.columns:
        raise KeyError(
            "Column 'Electricity_Consumed' is required for Prophet formatting."
        )

    frame = df[["Timestamp", "Electricity_Consumed"]].copy()
    frame = frame.rename(
        columns={"Timestamp": "ds", "Electricity_Consumed": "y"}
    )
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame


def _prophet_future_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    """Build a Prophet future frame from test ``Timestamp`` values."""
    if "Timestamp" not in test_df.columns:
        raise KeyError("Column 'Timestamp' is required for Prophet forecasting.")

    future = test_df[["Timestamp"]].copy()
    future = future.rename(columns={"Timestamp": "ds"})
    future["ds"] = pd.to_datetime(future["ds"])
    return future


def train_prophet_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Fit Facebook Prophet on train and forecast the test window.

    Prophet expects columns named ``ds`` (datetime) and ``y`` (target). This
    function copies ``train_df`` / ``test_df`` and maps ``Timestamp`` → ``ds``
    and ``Electricity_Consumed`` → ``y`` without mutating the caller's frames.

    Only ``train_df`` is used for fitting. Predictions are produced for every
    row in ``test_df`` in chronological order.

    Args:
        train_df: Chronological training split with ``Timestamp`` and
            ``Electricity_Consumed``.
        test_df: Chronological test split with ``Timestamp`` (defines forecast
            horizon).

    Returns:
        ``numpy.ndarray`` of shape ``(len(test_df),)`` with Prophet ``yhat``
        forecasts aligned to ``test_df`` row order.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If either split is empty.
    """
    from prophet import Prophet

    if len(train_df) == 0:
        raise ValueError("train_df must be non-empty.")
    if len(test_df) == 0:
        raise ValueError("test_df must be non-empty.")

    train_prophet = _prophet_train_frame(train_df)
    future = _prophet_future_frame(test_df)

    model = Prophet()
    model.fit(train_prophet)
    forecast = model.predict(future)

    return forecast["yhat"].to_numpy(dtype=np.float64)

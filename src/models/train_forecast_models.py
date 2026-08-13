"""Phase 3 forecasting model trainers.

Naive seasonal baseline, Prophet, XGBoost, and LSTM trainers. Advanced models
must beat the naive floor on the same held-out test window to be useful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from xgboost import XGBRegressor

    from src.models.lstm_model import EnergyLSTM


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
    """Copy ``Timestamp`` / ``Electricity_Consumed`` and rename to Prophet ``ds`` / ``y``.

    Args:
        df: Split DataFrame with ``Timestamp`` and ``Electricity_Consumed``.

    Returns:
        New DataFrame with columns ``ds`` (datetime) and ``y`` (float target).

    Raises:
        KeyError: If required columns are missing.
    """
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
    """Build a Prophet future frame from test ``Timestamp`` values.

    Args:
        test_df: Chronological test split containing ``Timestamp``.

    Returns:
        DataFrame with a single ``ds`` datetime column for ``Prophet.predict``.

    Raises:
        KeyError: If ``Timestamp`` is missing.
    """
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


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBRegressor:
    """Fit an XGBoost regressor with validation monitoring.

    Uses ``eval_set`` on the chronological validation split so training loss
    can be tracked without touching the held-out test window.

    Args:
        X_train: Training feature matrix (tabular predictors only).
        y_train: Training target aligned row-for-row with ``X_train``.
        X_val: Validation feature matrix with the same columns as ``X_train``.
        y_val: Validation target aligned row-for-row with ``X_val``.

    Returns:
        Fitted ``xgboost.XGBRegressor`` instance.

    Raises:
        ValueError: If train/val frames are empty or X/y lengths mismatch.
    """
    from xgboost import XGBRegressor

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("X_train and X_val must be non-empty.")
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("y_train and y_val must be non-empty.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train rows ({len(X_train)}) must match y_train ({len(y_train)})."
        )
    if len(X_val) != len(y_val):
        raise ValueError(
            f"X_val rows ({len(X_val)}) must match y_val ({len(y_val)})."
        )

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def make_lstm_dataloader(
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 32,
    consumption_index: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap LSTM sequence arrays in a PyTorch ``DataLoader``.

    Converts ``X`` and ``y`` to ``float32`` tensors and batches them for
    ``train_lstm_model``. When ``y`` has shape ``(N, n_features)``, only the
    consumption column at ``consumption_index`` is used as the scalar target
    (default index ``0`` = ``Electricity_Consumed`` in the prep feature order).

    Args:
        X: Sequence features with shape ``(N, seq_len, n_features)``.
        y: Targets with shape ``(N,)`` or ``(N, n_features)``.
        batch_size: Mini-batch size. Defaults to ``32``.
        consumption_index: Column index for consumption when ``y`` is 2D.
        shuffle: Whether to shuffle batches. Defaults to ``False`` to preserve
            chronological order within a split.

    Returns:
        ``torch.utils.data.DataLoader`` yielding ``(batch_X, batch_y)`` tuples.

    Raises:
        ValueError: If ``X`` and ``y`` lengths mismatch or arrays are empty.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.size == 0 or y_arr.size == 0:
        raise ValueError("X and y must be non-empty.")
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"X samples ({len(X_arr)}) must match y samples ({len(y_arr)})."
        )
    if y_arr.ndim == 2:
        y_arr = y_arr[:, consumption_index]
    elif y_arr.ndim != 1:
        raise ValueError(
            f"y must be 1D or 2D; got ndim={y_arr.ndim}."
        )

    X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_arr, dtype=torch.float32).reshape(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm_model(
    model: EnergyLSTM | Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> EnergyLSTM | Module:
    """Train an ``EnergyLSTM`` with Adam and MSE loss.

    Runs a standard PyTorch epoch loop: zero gradients, forward pass, loss,
    backward pass, optimizer step. Prints mean training and validation loss
    per epoch to monitor overfitting.

    Args:
        model: ``EnergyLSTM`` instance (or compatible ``nn.Module`` returning
            shape ``(batch,)`` predictions).
        train_loader: ``DataLoader`` of ``(batch_X, batch_y)`` training batches.
        val_loader: ``DataLoader`` of validation batches.
        epochs: Number of full passes over the training loader. Defaults to ``20``.
        learning_rate: Adam learning rate. Defaults to ``1e-3``.

    Returns:
        The same ``model`` instance, trained in place.

    Raises:
        ValueError: If ``epochs < 1`` or either loader is empty.
    """
    import torch
    import torch.nn as nn

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    if len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch.")
    if len(val_loader) == 0:
        raise ValueError("val_loader must contain at least one batch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item()
                val_batches += 1

        train_loss = train_loss_sum / train_batches
        val_loss = val_loss_sum / val_batches
        print(
            f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f} — "
            f"val_loss: {val_loss:.6f}"
        )

    return model


def predict_lstm(
    model: EnergyLSTM | Module,
    test_loader: DataLoader,
) -> np.ndarray:
    """Run inference on a test ``DataLoader`` and return flat NumPy predictions.

    Sets ``model.eval()`` and runs under ``torch.no_grad()``. Outputs are
    detached from the computation graph and moved to CPU for sklearn metrics.

    Args:
        model: Trained ``EnergyLSTM`` (or compatible module on the target device).
        test_loader: ``DataLoader`` of ``(batch_X, batch_y)`` test batches.
            Targets in ``batch_y`` are ignored; callers align actuals separately.

    Returns:
        1-D ``float`` array of predictions in loader order (length equals the
        number of test samples when ``shuffle=False``).

    Raises:
        ValueError: If ``test_loader`` is empty.
    """
    import torch

    if len(test_loader) == 0:
        raise ValueError("test_loader must contain at least one batch.")

    device = next(model.parameters()).device
    model.eval()
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            chunks.append(outputs.detach().cpu().numpy())

    return np.concatenate(chunks).reshape(-1)

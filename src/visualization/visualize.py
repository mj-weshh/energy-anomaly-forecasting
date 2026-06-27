"""Modular visualization utilities for smart meter exploratory data analysis.

All functions accept a pre-loaded DataFrame and return matplotlib Figure or Axes
objects for use in notebooks and scripts. This module does not read CSV files;
load data via :mod:`src.data.ingest_data` first.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_NUMERIC_FEATURES: list[str] = [
    "Electricity_Consumed",
    "Temperature",
    "Humidity",
    "Wind_Speed",
]

DEFAULT_HEATMAP_FEATURES: list[str] = [
    "Electricity_Consumed",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Avg_Past_Consumption",
]

DAY_ORDER: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day and day-of-week columns from ``Timestamp``.

    Args:
        df: DataFrame with a parsed ``Timestamp`` column.

    Returns:
        Copy of ``df`` with added columns: ``hour``, ``day_of_week``,
        ``day_name`` (ordered categorical), and ``is_weekend``.

    Raises:
        KeyError: If ``Timestamp`` is not present in ``df``.
    """
    if "Timestamp" not in df.columns:
        raise KeyError("Required column 'Timestamp' not found in DataFrame.")

    result = df.copy()
    result["hour"] = result["Timestamp"].dt.hour
    result["day_of_week"] = result["Timestamp"].dt.dayofweek
    result["day_name"] = pd.Categorical(
        result["Timestamp"].dt.day_name(),
        categories=DAY_ORDER,
        ordered=True,
    )
    result["is_weekend"] = result["day_of_week"] >= 5
    return result


def plot_feature_histograms(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] = (12, 8),
    bins: int = 30,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot histograms with KDE overlays for numeric features.

    Args:
        df: Loaded smart meter DataFrame.
        features: Columns to plot. Defaults to consumption and weather features.
        figsize: Figure size when creating a new figure.
        bins: Number of histogram bins.
        ax: Optional matplotlib Axes (unused; subplots are always created).

    Returns:
        matplotlib Figure containing one subplot per feature.

    Raises:
        ValueError: If any requested feature is missing from ``df``.
    """
    features = list(features or DEFAULT_NUMERIC_FEATURES)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    n = len(features)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, feature in enumerate(features):
        sns.histplot(
            df[feature],
            kde=True,
            bins=bins,
            ax=axes_flat[idx],
            color="steelblue",
            edgecolor="white",
        )
        axes_flat[idx].set_title(f"Distribution of {feature}")
        axes_flat[idx].set_xlabel(feature)
        axes_flat[idx].set_ylabel("Count")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_hourly_load_profile(
    df: pd.DataFrame,
    target: str = "Electricity_Consumed",
    *,
    figsize: tuple[float, float] = (12, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot consumption distribution by hour of day (daily load profile).

    Args:
        df: DataFrame with ``hour`` column (use :func:`add_temporal_features` first).
        target: Numeric column to aggregate by hour.
        figsize: Figure size when creating a new figure.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure with the hourly boxplot.

    Raises:
        KeyError: If ``hour`` or ``target`` is missing from ``df``.
    """
    if "hour" not in df.columns:
        raise KeyError("Column 'hour' not found. Call add_temporal_features() first.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.boxplot(
        data=df,
        x="hour",
        y=target,
        ax=ax,
        color="steelblue",
        fliersize=2,
    )
    ax.set_title("Daily Load Profile — Consumption by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(target)
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig


def plot_weekly_load_profile(
    df: pd.DataFrame,
    target: str = "Electricity_Consumed",
    *,
    figsize: tuple[float, float] = (10, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot consumption distribution by day of week (weekly seasonality).

    Args:
        df: DataFrame with ``day_name`` column (use :func:`add_temporal_features` first).
        target: Numeric column to aggregate by day.
        figsize: Figure size when creating a new figure.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure with the weekly boxplot.

    Raises:
        KeyError: If ``day_name`` or ``target`` is missing from ``df``.
    """
    if "day_name" not in df.columns:
        raise KeyError(
            "Column 'day_name' not found. Call add_temporal_features() first."
        )
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.boxplot(
        data=df,
        x="day_name",
        y=target,
        order=DAY_ORDER,
        ax=ax,
        color="steelblue",
        fliersize=2,
    )
    ax.set_title("Weekly Load Profile — Consumption by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel(target)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a Pearson correlation heatmap for numeric features.

    Args:
        df: Loaded smart meter DataFrame.
        features: Columns to include. Defaults to consumption, weather, and
            ``Avg_Past_Consumption``.
        figsize: Figure size when creating a new figure.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure with annotated correlation heatmap.

    Raises:
        ValueError: If any requested feature is missing from ``df``.
    """
    features = list(features or DEFAULT_HEATMAP_FEATURES)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    corr = df[features].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_consumption_timeseries(
    df: pd.DataFrame,
    target: str = "Electricity_Consumed",
    *,
    window: int = 48,
    figsize: tuple[float, float] = (14, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot raw consumption time series with a rolling average overlay.

    Args:
        df: DataFrame with ``Timestamp`` and ``target`` columns.
        target: Numeric column to plot.
        window: Rolling window size in rows. Default 48 = 24 hours at 30-min intervals.
        figsize: Figure size when creating a new figure.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure with raw series and rolling mean.

    Raises:
        KeyError: If ``Timestamp`` or ``target`` is missing from ``df``.
    """
    if "Timestamp" not in df.columns:
        raise KeyError("Required column 'Timestamp' not found in DataFrame.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rolling = df[target].rolling(window=window, min_periods=1).mean()

    ax.plot(
        df["Timestamp"],
        df[target],
        alpha=0.35,
        linewidth=0.8,
        label="30-min readings",
        color="steelblue",
    )
    ax.plot(
        df["Timestamp"],
        rolling,
        linewidth=2,
        label=f"{window}-period rolling mean ({window * 30 / 60:.0f}h)",
        color="darkorange",
    )
    ax.set_title("Electricity Consumption Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(target)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

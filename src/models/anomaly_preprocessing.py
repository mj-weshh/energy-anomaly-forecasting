"""Train-fitted preprocessing for anomaly detection tuning.

Fits ``StandardScaler`` and optional per-hour consumption z-scores on the
training split only to avoid leakage during temporal validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WEATHER_COLUMNS = ("Temperature", "Humidity", "Wind_Speed")


@dataclass
class AnomalyPreprocessor:
    """Scale features and append train-fitted hourly consumption z-scores."""

    add_hour_zscore: bool = True
    scaler: StandardScaler = field(default_factory=StandardScaler)
    hour_stats: dict[int, tuple[float, float]] = field(default_factory=dict)
    feature_columns_: list[str] = field(default_factory=list)
    fitted_: bool = False

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        hours: pd.Series,
        consumption: pd.Series,
        train_mask: np.ndarray,
    ) -> AnomalyPreprocessor:
        """Fit scaler and hourly stats on training rows only."""
        if train_mask.shape[0] != len(feature_matrix):
            raise ValueError("train_mask length must match feature_matrix rows.")

        self.feature_columns_ = list(feature_matrix.columns)
        train_X = feature_matrix.iloc[train_mask]
        self.scaler.fit(train_X)

        if self.add_hour_zscore:
            train_hours = hours.iloc[train_mask]
            train_consumption = consumption.iloc[train_mask]
            self.hour_stats = {}
            for hour in range(24):
                mask = train_hours == hour
                if not mask.any():
                    continue
                values = train_consumption.loc[mask].astype(float)
                std = float(values.std(ddof=0))
                self.hour_stats[hour] = (float(values.mean()), std if std > 0 else 1.0)

        self.fitted_ = True
        return self

    def transform(
        self,
        feature_matrix: pd.DataFrame,
        hours: pd.Series,
        consumption: pd.Series,
    ) -> np.ndarray:
        """Return scaled feature matrix with optional hourly z-score column."""
        if not self.fitted_:
            raise ValueError("AnomalyPreprocessor must be fitted before transform.")

        ordered = feature_matrix[self.feature_columns_]
        scaled = self.scaler.transform(ordered)

        if not self.add_hour_zscore:
            return scaled

        zscores = np.zeros(len(feature_matrix), dtype=float)
        for idx, (hour, value) in enumerate(zip(hours, consumption, strict=True)):
            if hour in self.hour_stats:
                mean, std = self.hour_stats[int(hour)]
                zscores[idx] = (float(value) - mean) / std
        return np.column_stack([scaled, zscores])

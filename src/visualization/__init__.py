"""Modular visualization utilities for smart meter exploratory data analysis."""

from src.visualization.visualize import (  # noqa: F401
    add_plot_temporal_features,
    plot_anomaly_label_distribution,
    plot_correlation_heatmap,
    plot_feature_histograms,
    plot_hourly_load_profile,
    plot_weekly_load_profile,
)

# Backward-compatible alias for EDA scripts written before the rename.
add_temporal_features = add_plot_temporal_features

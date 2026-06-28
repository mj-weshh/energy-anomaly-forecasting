"""Export EDA plot PNGs to docs/assets/eda/ for MkDocs documentation.

Run from repository root::

    python scripts/export_eda_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.visualization.visualize import (  # noqa: E402
    add_temporal_features,
    plot_anomaly_label_distribution,
    plot_consumption_timeseries,
    plot_correlation_heatmap,
    plot_feature_histograms,
    plot_hourly_load_profile,
    plot_weekly_load_profile,
)

OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "eda"
DPI = 150


def _save(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _print_metrics(df) -> None:
    hourly = df.groupby("hour", observed=True)["Electricity_Consumed"].mean()
    peak_hour = int(hourly.idxmax())
    peak_mean = hourly.max()
    top3 = hourly.nlargest(3)

    weekday_mask = df["day_name"].isin(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    weekend_mask = df["day_name"].isin(["Saturday", "Sunday"])
    weekday_mean = df.loc[weekday_mask, "Electricity_Consumed"].mean()
    weekend_mean = df.loc[weekend_mask, "Electricity_Consumed"].mean()
    dow = df.groupby("day_name", observed=True)["Electricity_Consumed"].mean()

    corr = df[
        [
            "Electricity_Consumed",
            "Temperature",
            "Humidity",
            "Wind_Speed",
            "Avg_Past_Consumption",
        ]
    ].corr()["Electricity_Consumed"]

    label_counts = df["Anomaly_Label"].value_counts()
    imbalance = label_counts["Normal"] / label_counts["Abnormal"]

    print("EDA metrics (for documentation verification)")
    print(f"Peak hour: {peak_hour} (mean {peak_mean:.3f})")
    print(
        "Top-3 hours: "
        + ", ".join(f"{int(h)}->{v:.3f}" for h, v in top3.items())
    )
    print(f"Weekday mean: {weekday_mean:.3f} | Weekend mean: {weekend_mean:.3f}")
    print(
        f"Friday: {dow['Friday']:.3f} | Wednesday: {dow['Wednesday']:.3f}"
    )
    print(
        "Correlations: "
        f"Avg_Past_Consumption {corr['Avg_Past_Consumption']:+.3f}, "
        f"Wind {corr['Wind_Speed']:+.3f}, "
        f"Humidity {corr['Humidity']:+.3f}, "
        f"Temperature {corr['Temperature']:+.3f}"
    )
    print(
        "Anomaly labels: "
        f"Normal {label_counts['Normal']} ({label_counts['Normal'] / len(df) * 100:.0f}%), "
        f"Abnormal {label_counts['Abnormal']} ({label_counts['Abnormal'] / len(df) * 100:.0f}%), "
        f"ratio {imbalance:.1f}:1"
    )


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = add_temporal_features(load_smart_meter_data(csv_path))

    saved: list[Path] = []
    saved.append(_save(plot_feature_histograms(df), "feature-distributions.png"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_hourly_load_profile(df, ax=axes[0])
    plot_weekly_load_profile(df, ax=axes[1])
    fig.tight_layout()
    saved.append(_save(fig, "load-profiles.png"))

    saved.append(_save(plot_correlation_heatmap(df), "correlation-heatmap.png"))
    saved.append(
        _save(plot_anomaly_label_distribution(df), "anomaly-label-distribution.png")
    )
    saved.append(
        _save(plot_consumption_timeseries(df, window=48), "consumption-timeseries.png")
    )

    print(f"\nLoaded: {csv_path}")
    print(f"Exported {len(saved)} PNGs to {OUTPUT_DIR}:")
    for path in saved:
        print(f"  - {path.relative_to(REPO_ROOT)}")

    print()
    _print_metrics(df)


if __name__ == "__main__":
    main()

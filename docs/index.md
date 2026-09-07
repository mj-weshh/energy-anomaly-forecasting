# Energy Anomaly Forecasting

This project turns **smart-meter electricity readings** into a clean timeline, finds unusual consumption, and forecasts what comes next. It uses a public [Kaggle smart-meter dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) (5,000 half-hour rows)—no proprietary utility data.

!!! success "In one minute"

    - **Goal:** Detect odd meter readings, repair the timeline, then forecast demand.
    - **Status:** End-to-end pipeline is complete (ingest → detect → clean → forecast → export).
    - **Headline:** On the default forecast comparison, **Prophet** has the lowest average error; all advanced models beat a simple “same time yesterday” baseline.
    - **Run it:** [Getting Started](getting-started.md) · full CLI: [E2E Pipeline](e2e-pipeline.md)
    - **Terms:** [Glossary](glossary.md)

## How to read these docs

| If you are… | Start here |
|-------------|------------|
| New to the project | This page → [Getting Started](getting-started.md) → [E2E Pipeline](e2e-pipeline.md) |
| Want results, not code | [Forecast Model Comparison](forecast-model-comparison.md) · [Forecasting Research](forecasting-research.md) |
| Implementing or reviewing code | [Architecture](architecture.md) · collapsible **Technical deep dive** blocks on each page |

## What we built (three phases)

| Phase | Plain English | Details |
|-------|---------------|---------|
| **1 — Understand the data** | Load and check the CSV; explore daily patterns and what correlates with consumption. | [EDA Insights](eda-insights.md) · [Data Schema](data-schema.md) |
| **2 — Find and fix anomalies** | Flag unusual readings (without using the label to train), then fill gaps so the series stays continuous. | [Anomaly Detection](anomaly-detection.md) · [Clean Dataset](clean-data.md) |
| **3 — Forecast** | Split time in order (past → future), compare four models, ship one CLI and teaching materials. | [Forecast Model Comparison](forecast-model-comparison.md) · [E2E Pipeline](e2e-pipeline.md) |

All work uses public data only.

## Results at a glance

**Forecasting** (lower error is better): a simple seasonal baseline sits near MAE **0.171** / RMSE **0.214**. **Prophet** leads under the default run (about **0.121** / **0.149**). Full table and chart: [Forecast Model Comparison](forecast-model-comparison.md). What that means: [Forecasting Research](forecasting-research.md).

**Anomaly detection:** Isolation Forest is the practical workhorse for cleaning. Research tuning improves held-out scores versus the conservative production-style baseline; the default clean file stays on the conservative recipe until leadership reviews alternatives. Details: [Anomaly Detection](anomaly-detection.md) · [Anomaly Tuning Results](anomaly-tuning-results.md).

## Where to go next

| Goal | Page |
|------|------|
| Install and run | [Getting Started](getting-started.md) |
| See how folders and data flow | [Architecture](architecture.md) |
| Run the full pipeline CLI | [E2E Pipeline](e2e-pipeline.md) |
| Compare forecast models | [Forecast Model Comparison](forecast-model-comparison.md) |
| Teaching notebook | [Forecasting Tutorial](forecasting-tutorial.md) |
| Definitions | [Glossary](glossary.md) |

```bash
python -m src.data.ingest_data
```

Expect shape `(5000, 7)`, zero nulls, and continuity **PASS**.

??? info "Technical deep dive"

    Phase 1 = ingest + EDA. Phase 2 = features + unsupervised detection + clean artifact. Phase 3 = forecast ladder via `compare_forecasts.py` and consolidating `main.py` (ingest → detect → clean → forecast → metrics → CSV). Fair-comparison anomaly metrics and research extensions: [Anomaly Tuning Results](anomaly-tuning-results.md).

## License

[MIT License](../LICENSE).

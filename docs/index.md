# Energy Anomaly Forecasting

Open-source machine learning project for **energy consumption anomaly detection** and **time-series forecasting**, built on the public [Kaggle Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).

## Mission

Develop reproducible pipelines to:

1. Ingest and validate 30-minute smart meter time-series data
2. Detect anomalous consumption patterns (Phase 2)
3. Forecast future energy demand (Phase 3)

All work uses publicly available data only. No proprietary systems or datasets are referenced.

## Current Status

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 Week 1 | Environment setup, data ingestion, schema validation | **Complete** |
| Phase 1 Week 2 | Exploratory data analysis and load profiling | **Complete** |
| Phase 2 Week 3 | Feature engineering (temporal + rolling features) | **Complete** |
| Phase 2 Week 4 | Anomaly detection (IF + DBSCAN baselines) | **Complete** |
| Phase 3 | Time-series forecasting (XGBoost, LSTM) | Planned |

### Phase 1 Week 2 highlights

- Peak mean consumption at **02:00** (hour 2); modest diurnal variation on normalized data
- **Weak weather correlation** with consumption (|r| &lt; 0.01 for temperature, humidity, wind)
- Strongest linear predictor: `Avg_Past_Consumption` (**r = +0.317**)
- Anomaly label baseline: **5% Abnormal** (250 / 5,000 rows; 19:1 imbalance)

![Feature correlation heatmap from Phase 1 Week 2 EDA](assets/eda/correlation-heatmap.png)

Full analysis: [EDA Insights](eda-insights.md)

### Phase 2 Week 4 highlights

- Isolation Forest baseline: **F1 = 0.331** on 4,953 eval rows (labels excluded from training)
- DBSCAN baseline + 12-combo grid search via `scripts/tune_dbscan.py` — best F1 = **0.125** at `eps=0.5`, `min_samples=5`
- Unified `detect_anomalies()` router for both Isolation Forest and DBSCAN
- **IF leads on F1** on this coarse grid; DBSCAN over-flags at most settings
- Clean dataset pipeline (Day 3): IF + time interpolation → `data/processed/clean_smart_meter_data.csv` (**5000 rows preserved**)
- `Anomaly_Label` used for evaluation only — never for model fitting

Full reports: [Anomaly Detection](anomaly-detection.md) · [Clean Dataset](clean-data.md)

## Documentation

| Document | Purpose |
|----------|---------|
| [Getting Started](getting-started.md) | Install dependencies and run ingestion locally, on Colab, or on Kaggle |
| [Data Schema](data-schema.md) | Formal data dictionary for `smart_meter_data.csv` |
| [Verification Report](verification-report.md) | Evidence that Phase 1 Week 1 acceptance criteria are met |
| [Architecture](architecture.md) | Repository layout, data flow, and design decisions |
| [EDA Insights](eda-insights.md) | Phase 1 Week 2 exploratory analysis findings with figures |
| [Phase 2 Strategy](phase2-strategy.md) | Anomaly detection planning grounded in Phase 1 EDA |
| [Feature Engineering](feature-engineering.md) | Phase 2 Week 3 temporal features, rolling metrics, and verification |
| [Anomaly Detection](anomaly-detection.md) | Phase 2 Week 4 IF + DBSCAN baselines, grid search, and model comparison |
| [Clean Dataset](clean-data.md) | Phase 2 Week 4 Day 3 anomaly masking, interpolation, and Phase 3 artifact |

## Quick Command

```bash
python -m src.data.ingest_data
```

Expected outcome: schema summary with shape `(5000, 7)`, zero nulls, and a continuity check **PASS**.

## License

This project is released under the [MIT License](../LICENSE).

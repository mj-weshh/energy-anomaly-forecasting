# Energy Anomaly Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-green.svg)](#verified-results)

Open-source machine learning project for **energy consumption anomaly detection** and **time-series forecasting**, built entirely on the public [Kaggle Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).

**Executive summary:** This project turns smart-meter data into a reliable timeline for analysis and forecasting. Phases 1–2 are complete (detect + clean). Phase 3 Week 6 Day 1–2 adds a clean-state gate, chronological 70/15/15 split, and a naive seasonal forecast floor (example test MAE ≈ **0.171**, RMSE ≈ **0.214**). **Production cleaning is unchanged** (~248 corrected intervals). Full docs: [docs site](docs/index.md) · [Forecasting Baseline](docs/forecasting-baseline.md) · [Glossary](docs/glossary.md).

---

## Overview

This repository implements a phased ML pipeline:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1 Week 1** | Environment setup, data ingestion, schema validation | **Complete** |
| **Phase 1 Week 2** | Exploratory data analysis and load profiling | **Complete** |
| **Phase 2 Week 3** | Feature engineering (temporal + rolling metrics) | **Complete** |
| **Phase 2 Week 4** | Anomaly detection (IF + DBSCAN baselines) | **Complete** |
| **Phase 3 Week 6 Day 1–2** | Forecasting foundation (gate, split, metrics, naive baseline) | **Complete** |
| **Phase 3 (next)** | Prophet/ARIMA → XGBoost → LSTM | Planned |

All work uses publicly available data. No proprietary datasets or systems are referenced.

---

## Quick Start

```bash
git clone https://github.com/mj-weshh/energy-anomaly-forecasting.git
cd energy-anomaly-forecasting

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python -m src.data.ingest_data
```

Expected output: schema summary with shape `(5000, 7)`, zero nulls, and continuity check **PASS**.

---

## Verified Results

Phase 1, Week 1 acceptance criteria — verified locally on `smart_meter_data.csv`:

| Check | Result |
|-------|--------|
| Shape | `(5000, 7)` |
| Null values | 0 across all columns |
| `Timestamp` dtype | `datetime64[ns]` |
| Sampling frequency | 30 minutes |
| Date range | `2024-01-01` → `2024-04-14` |
| Continuity | **PASS** — no gaps, duplicates, or irregular intervals |

![Schema summary verification](docs/assets/schema-summary.png)

Full evidence: [Verification Report](docs/verification-report.md)

---

## EDA Results (Phase 1, Week 2)

Key findings from exploratory analysis on the same dataset:

| Finding | Result |
|---------|--------|
| Peak mean consumption hour | **02:00** |
| Weather correlation with consumption | Negligible (|r| &lt; 0.01) |
| Strongest linear predictor | `Avg_Past_Consumption` (r = +0.317) |
| Anomaly label baseline | **5% Abnormal** (250 / 5,000) |

![Daily and weekly load profiles](docs/assets/eda/load-profiles.png)

Full report with all figures: [EDA Insights](docs/eda-insights.md)

Regenerate doc figures: `python scripts/export_eda_assets.py` (EDA) · `python scripts/generate_mermaid_assets.py` (architecture PNGs)

---

## Project Structure

```
energy-anomaly-forecasting/
├── data/
│   ├── raw/                        # Canonical raw data location (optional)
│   └── processed/                  # Generated clean CSV (gitignored)
├── docs/                           # Documentation (MkDocs source)
│   └── assets/                     # Verification screenshots and EDA figures
│       └── eda/                    # Exported Phase 1 Week 2 plots (PNG)
├── notebooks/
│   ├── 01_data_ingestion_and_schema_check.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_anomaly_detection.ipynb
├── scripts/
│   ├── export_eda_assets.py        # Regenerate EDA doc figures
│   ├── generate_mermaid_assets.py  # Regenerate architecture PNGs (mermaid.ink)
│   ├── verify_features.py          # Sanity-check engineered features
│   ├── test_isolation_forest.py    # Isolation Forest baseline + evaluation
│   ├── tune_isolation_forest.py    # Enhanced IF hyperparameter + threshold tuning
│   ├── tune_dbscan.py              # DBSCAN hyperparameter grid search
│   ├── tune_ensemble.py            # IF + DBSCAN ensemble comparison
│   ├── compare_anomaly_models.py   # Legacy vs enhanced research dashboard
│   ├── analyze_detection_errors.py # Legacy IF hourly FP analysis
│   ├── compare_clean_artifacts.py  # Diff legacy vs research clean CSVs
│   ├── tune_isolation_forest_by_segment.py  # Per-hour/weekend test F1
│   ├── generate_clean_data.py      # Generate Phase 3 clean dataset artifact
│   ├── verify_phase2_state.py      # Phase 3 gate: clean CSV continuity / NaNs
│   └── evaluate_naive_baseline.py  # Score naive seasonal forecast on test set
├── src/
│   ├── data/
│   │   ├── ingest_data.py          # Canonical ingestion module
│   │   ├── clean_data.py           # Anomaly masking and interpolation
│   │   └── make_forecast_dataset.py # Chronological train/val/test split
│   ├── features/
│   │   └── build_features.py       # Temporal + rolling feature engineering
│   ├── models/
│   │   ├── evaluate_models.py      # Imbalance-aware anomaly evaluation
│   │   ├── evaluate_forecast.py    # Forecast MAE / RMSE / MAPE
│   │   ├── train_anomaly_models.py # Unsupervised anomaly training
│   │   ├── train_forecast_models.py # Naive seasonal baseline (+ later models)
│   │   ├── anomaly_preprocessing.py # Train-fitted scaling for tuning
│   │   ├── tuning_utils.py         # Temporal splits and threshold search
│   │   └── anomaly_config.py       # Research-tuned hyperparameters
│   └── visualization/
│       └── visualize.py            # EDA plotting functions
├── Smart Meter Electricity Consumption Dataset/
│   └── smart_meter_data.csv
├── requirements.txt
├── mkdocs.yml
└── README.md
```

---

## Dataset

**Source:** [Kaggle — Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) by [ziya07](https://www.kaggle.com/ziya07)

| Property | Value |
|----------|-------|
| Filename | `smart_meter_data.csv` |
| Rows | 5,000 |
| Interval | 30 minutes |
| Columns | 7 (timestamp, 5 features, 1 label) |

Place the CSV in `data/raw/` or keep it in `Smart Meter Electricity Consumption Dataset/`. The ingestion script discovers it automatically.

Schema reference: [Data Schema](docs/data-schema.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Local, Colab, and Kaggle setup |
| [Data Schema](docs/data-schema.md) | Formal data dictionary |
| [Verification Report](docs/verification-report.md) | Phase 1 Week 1 QA evidence |
| [EDA Insights](docs/eda-insights.md) | Phase 1 Week 2 findings with figures |
| [Architecture](docs/architecture.md) | Repository layout and data flow |
| [Phase 2 Strategy](docs/phase2-strategy.md) | Anomaly detection planning grounded in Phase 1 EDA |
| [Feature Engineering](docs/feature-engineering.md) | Phase 2 Week 3 temporal features and rolling metrics |
| [Anomaly Detection](docs/anomaly-detection.md) | Phase 2 Week 4 IF + DBSCAN baselines, grid search, model comparison, and educational notebook |
| [Anomaly Tuning Results](docs/anomaly-tuning-results.md) | Phase 2 research tuning report — enhanced features, temporal splits, fair comparison |
| [Clean Dataset](docs/clean-data.md) | Phase 2 Week 4 Day 3 anomaly masking, interpolation, and Phase 3 artifact |
| [Forecasting Baseline](docs/forecasting-baseline.md) | Phase 3 Week 6 Day 1–2 gate, chronological split, metrics, naive floor |
| [Phase 3 Strategy](docs/phase3-strategy.md) | Forecasting planning — model ladder and evaluation protocol |
| [Glossary](docs/glossary.md) | Shared plain-English and technical term definitions |

### Build docs site locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve    # http://127.0.0.1:8000
mkdocs build    # output to site/
```

---

## Usage

### CLI

```bash
python -m src.data.ingest_data
python scripts/export_eda_assets.py
python scripts/generate_mermaid_assets.py
python scripts/verify_features.py
python scripts/test_isolation_forest.py
python scripts/tune_dbscan.py
python scripts/tune_isolation_forest.py
python scripts/tune_ensemble.py
python scripts/compare_anomaly_models.py
python scripts/analyze_detection_errors.py
python scripts/tune_isolation_forest_by_segment.py
python scripts/generate_clean_data.py
python scripts/generate_clean_data.py --profile legacy_threshold
python scripts/generate_clean_data.py --profile enhanced
python scripts/compare_clean_artifacts.py
python scripts/verify_phase2_state.py
python -m src.data.make_forecast_dataset
python scripts/evaluate_naive_baseline.py
```

### Python API

```python
from src.data.ingest_data import find_dataset_csv, load_smart_meter_data, get_project_root

csv_path = find_dataset_csv(get_project_root())
df = load_smart_meter_data(csv_path)
print(df.shape)  # (5000, 7)
```

### Feature Engineering (Phase 2)

```python
from src.features.build_features import add_temporal_features, add_rolling_metrics, build_all_features

df = build_all_features(df)  # or: add_rolling_metrics(add_temporal_features(df))
print(df.shape)  # (5000, 15) — adds hour, day_of_week, month, is_weekend,
                 # and 3h/24h rolling mean + std over Electricity_Consumed
```

### Anomaly Detection (Phase 2)

```python
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import detect_anomalies, train_dbscan, train_isolation_forest

df_feat = build_all_features(df)

# Unified router
model, predictions = detect_anomalies(df_feat, model_type="isolation_forest")
model, predictions = detect_anomalies(df_feat, model_type="dbscan", eps=0.5, min_samples=5)
```

### Clean Dataset (Phase 2 → Phase 3)

```python
from src.data.clean_data import generate_clean_dataset
from src.data.ingest_data import find_dataset_csv, get_project_root

generate_clean_dataset(
    str(find_dataset_csv(get_project_root())),
    "data/processed/clean_smart_meter_data.csv",
)
```

### Forecasting Baseline (Phase 3 Day 1–2)

```python
import pandas as pd
from src.data.make_forecast_dataset import time_series_split
from src.models.train_forecast_models import naive_seasonal_forecast
from src.models.evaluate_forecast import evaluate_forecast

df = pd.read_csv("data/processed/clean_smart_meter_data.csv", parse_dates=["Timestamp"])
train, val, test = time_series_split(df)
y_pred = naive_seasonal_forecast(
    train["Electricity_Consumed"].to_numpy(),
    test["Electricity_Consumed"].to_numpy(),
    seasonal_periods=48,
)
print(evaluate_forecast(test["Electricity_Consumed"].to_numpy(), y_pred))
```

Full notes: [Forecasting Baseline](docs/forecasting-baseline.md).

### Phase 2 research results (held-out test)

Production cleaning still uses legacy IF (full-dataset F1 **0.331**). Research tuning on the same 991-row test window (`scripts/compare_anomaly_models.py`):

| Model | Test F1 |
|-------|---------|
| Legacy IF (production params) | 0.340 |
| Legacy IF (val threshold) | 0.389 |
| Enhanced IF (tuned) | **0.460** |

Full methodology: [Anomaly Tuning Results](docs/anomaly-tuning-results.md) · metrics in `src/models/anomaly_config.py`.

### Notebooks

- [`notebooks/01_data_ingestion_and_schema_check.ipynb`](notebooks/01_data_ingestion_and_schema_check.ipynb) — ingestion and schema validation
- [`notebooks/02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) — Phase 1 Week 2 EDA
- [`notebooks/03_anomaly_detection.ipynb`](notebooks/03_anomaly_detection.ipynb) — Phase 2 Week 4 CMU tutorial: unsupervised detection, benchmark evaluation, and consumption interpolation

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Waweru Muhura

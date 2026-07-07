# Energy Anomaly Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-green.svg)](#verified-results)

Open-source machine learning project for **energy consumption anomaly detection** and **time-series forecasting**, built entirely on the public [Kaggle Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).

---

## Overview

This repository implements a phased ML pipeline:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1 Week 1** | Environment setup, data ingestion, schema validation | **Complete** |
| **Phase 1 Week 2** | Exploratory data analysis and load profiling | **Complete** |
| **Phase 2 Week 3** | Feature engineering (temporal + rolling metrics) | **Complete** |
| **Phase 2 Week 4** | Anomaly detection (IF + DBSCAN baselines) | **Complete** |
| **Phase 3** | Time-series forecasting (XGBoost, LSTM) | Planned |

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

Regenerate doc figures: `python scripts/export_eda_assets.py`

---

## Project Structure

```
energy-anomaly-forecasting/
├── data/raw/                       # Canonical raw data location (optional)
├── docs/                           # Documentation (MkDocs source)
│   └── assets/                     # Verification screenshots and EDA figures
│       └── eda/                    # Exported Phase 1 Week 2 plots (PNG)
├── notebooks/
│   ├── 01_data_ingestion_and_schema_check.ipynb
│   └── 02_exploratory_data_analysis.ipynb
├── scripts/
│   ├── export_eda_assets.py        # Regenerate EDA doc figures
│   ├── verify_features.py          # Sanity-check engineered features
│   ├── test_isolation_forest.py    # Isolation Forest baseline + evaluation
│   └── tune_dbscan.py              # DBSCAN hyperparameter grid search
├── src/
│   ├── data/
│   │   └── ingest_data.py          # Canonical ingestion module
│   ├── features/
│   │   └── build_features.py       # Temporal + rolling feature engineering
│   ├── models/
│   │   ├── evaluate_models.py      # Imbalance-aware evaluation metrics
│   │   └── train_anomaly_models.py # Unsupervised anomaly training
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
| [Anomaly Detection](docs/anomaly-detection.md) | Phase 2 Week 4 IF + DBSCAN baselines, grid search, and model comparison |

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
python scripts/verify_features.py
python scripts/test_isolation_forest.py
python scripts/tune_dbscan.py
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

### Notebooks

- [`notebooks/01_data_ingestion_and_schema_check.ipynb`](notebooks/01_data_ingestion_and_schema_check.ipynb) — ingestion and schema validation
- [`notebooks/02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) — Phase 1 Week 2 EDA

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Waweru Muhura

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
| **Phase 1** | Environment setup, data ingestion, schema validation, EDA | Ingestion **complete** |
| **Phase 2** | Unsupervised anomaly detection (Isolation Forest, DBSCAN) | Planned |
| **Phase 3** | Time-series forecasting (XGBoost, LSTM) | Planned |

All work uses publicly available data. No proprietary datasets or systems are referenced.

---

## Quick Start

```bash
git clone https://github.com/<your-org>/energy-anomaly-forecasting.git
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

## Project Structure

```
energy-anomaly-forecasting/
├── data/raw/                       # Canonical raw data location (optional)
├── docs/                           # Documentation (MkDocs source)
│   └── assets/                     # Verification screenshots
├── notebooks/
│   └── 01_data_ingestion_and_schema_check.ipynb
├── src/
│   └── data/
│       └── ingest_data.py          # Canonical ingestion module
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
| [Verification Report](docs/verification-report.md) | Phase 1 QA evidence |
| [Architecture](docs/architecture.md) | Repository layout and data flow |

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
```

### Python API

```python
from src.data.ingest_data import find_dataset_csv, load_smart_meter_data, get_project_root

csv_path = find_dataset_csv(get_project_root())
df = load_smart_meter_data(csv_path)
print(df.shape)  # (5000, 7)
```

### Notebook

Run [`notebooks/01_data_ingestion_and_schema_check.ipynb`](notebooks/01_data_ingestion_and_schema_check.ipynb) for an interactive version with Colab and Kaggle setup options.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Waweru Muhura

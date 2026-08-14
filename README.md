# Energy Anomaly Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MkDocs](https://img.shields.io/badge/docs-MkDocs%20Material-indigo.svg)](docs/index.md)
[![Phase 3](https://img.shields.io/badge/Phase%203-Complete-green.svg)](#results-at-a-glance)

Open-source ML pipeline for **smart-meter anomaly detection** and **short-horizon energy forecasting**, built on the public [Kaggle Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).

Developed for **Wawtex Solutions** and **CMU-Africa** handover — reproducible end-to-end from raw CSV to scored forecasts.

---

## Project summary

This repository turns 5,000 half-hourly meter readings into a clean timeline, flags anomalies without using the benchmark label for training, and forecasts consumption with a four-model ladder:

| Stage | What it does |
|-------|----------------|
| **Ingest & EDA** | Load, validate schema, profile load patterns |
| **Detect & clean** | Isolation Forest / DBSCAN → interpolate flagged intervals |
| **Forecast** | Naive → Prophet → XGBoost → LSTM on chronological splits |
| **Ship** | Root CLI (`main.py`), tutorial notebook, research write-up, MkDocs site |

**Default ladder winner (test MAE / RMSE):** Prophet ≈ **0.121 / 0.149**, beating the seasonal-naive floor (≈ **0.171 / 0.214**). Full narrative: [Forecasting Research](docs/forecasting-research.md).

**Documentation site (source):** start at [`docs/index.md`](docs/index.md) or run locally with MkDocs (below).

---

## Tech stack

| Area | Libraries |
|------|-----------|
| **Core data** | pandas, numpy |
| **Modeling** | scikit-learn, Prophet, XGBoost, PyTorch |
| **Visualization** | matplotlib, seaborn |
| **Notebooks** | Jupyter, ipykernel |
| **Docs & tests** | MkDocs Material, pytest |

Pinned minimums live in [`requirements.txt`](requirements.txt).

---

## How to run

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mj-weshh/energy-anomaly-forecasting.git
cd energy-anomaly-forecasting

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Put `smart_meter_data.csv` in either:

- `Smart Meter Electricity Consumption Dataset/smart_meter_data.csv`, or
- `data/raw/smart_meter_data.csv`

Download: [Kaggle dataset page](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).

### 4. Run the end-to-end pipeline

```bash
python main.py --model naive
```

This runs **ingest → feature engineering → anomaly detection → interpolation → chronological split → forecast → metrics → CSV export** (`data/processed/final_predictions.csv` by default).

Other models:

```bash
python main.py --model prophet
python main.py --model xgboost
python main.py --model lstm --epochs 20
python main.py --help
```

Optional flags: `--data_path`, `--save_clean_data`, `--output_path`.

### 5. Smoke-check ingestion (optional)

```bash
python -m src.data.ingest_data
```

Expect shape `(5000, 7)`, zero nulls, continuity **PASS**.

### 6. Browse the docs site (optional)

```bash
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Build static site with `mkdocs build`.

---

## Results at a glance

| Check | Result |
|-------|--------|
| Schema | `(5000, 7)`, 30-minute grid, `2024-01-01` → `2024-04-14` |
| Continuity | **PASS** — no gaps or duplicate timestamps |
| Anomaly label baseline | 5% Abnormal (benchmark only; models train unsupervised) |
| Production clean artifact | ~248 corrected intervals via `scripts/generate_clean_data.py` |
| Forecast ladder (test) | Naive → Prophet → XGBoost → LSTM; Prophet leads on default MAE/RMSE |

![Forecast model comparison (3-day window)](docs/assets/forecast_comparison.png)

Unified metrics table + figure:

```bash
python scripts/compare_forecasts.py
```

Tutorial path: [`notebooks/04_forecasting_tutorial.ipynb`](notebooks/04_forecasting_tutorial.ipynb) · [Forecasting Tutorial](docs/forecasting-tutorial.md)

---

## Repository layout

```
energy-anomaly-forecasting/
├── main.py                 # E2E CLI (ingest → forecast → metrics → CSV)
├── requirements.txt
├── mkdocs.yml
├── data/                   # raw/ + processed/ (generated artifacts)
├── docs/                   # MkDocs pages + assets/
├── notebooks/              # Ingestion, EDA, anomaly, forecasting tutorial
├── scripts/                # Evaluation, tuning, figure export helpers
├── src/
│   ├── data/               # Ingest, clean, chronological split
│   ├── features/           # Temporal, rolling, lags, LSTM sequences
│   ├── models/             # Anomaly + forecast trainers & metrics
│   └── visualization/      # EDA plotting helpers
└── tests/                  # pytest suite (CI)
```

Deep dive: [Architecture](docs/architecture.md).

---

## Documentation map

| Start here | Link |
|------------|------|
| Docs home | [docs/index.md](docs/index.md) |
| Setup detail | [Getting Started](docs/getting-started.md) |
| E2E CLI | [E2E Pipeline](docs/e2e-pipeline.md) |
| Model ladder | [Forecast Model Comparison](docs/forecast-model-comparison.md) |
| Research findings | [Forecasting Research](docs/forecasting-research.md) |
| Terms | [Glossary](docs/glossary.md) |

---

## Useful commands

```bash
# Phase 2 clean artifact for forecasting scripts
python scripts/generate_clean_data.py
python scripts/verify_phase2_state.py

# Individual forecast evaluators
python scripts/evaluate_naive_baseline.py
python scripts/evaluate_prophet.py
python scripts/evaluate_xgboost.py
python scripts/evaluate_lstm.py

# Regenerated doc figures
python scripts/export_eda_assets.py
python scripts/compare_forecasts.py
python scripts/export_xgboost_feature_importance.py

# Tests
pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE).

**Dataset:** [Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) (Kaggle / ziya07). Respect the dataset license on Kaggle when redistributing data files.

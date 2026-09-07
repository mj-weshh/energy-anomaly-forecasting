# Getting Started

Install the project, verify the dataset, and run the main pipeline.

!!! success "In one minute"

    - Clone, create a virtual environment, `pip install -r requirements.txt`.
    - Place `smart_meter_data.csv` (or use the bundled copy).
    - Smoke-check: `python -m src.data.ingest_data` → continuity **PASS**.
    - Full path: `python main.py --model naive` → see [E2E Pipeline](e2e-pipeline.md).
    - Terms: [Glossary](glossary.md).

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 or later |
| Git | Any recent version |
| pip | Bundled with Python |

---

## 1. Clone and set up

```bash
git clone https://github.com/mj-weshh/energy-anomaly-forecasting.git
cd energy-anomaly-forecasting

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Obtain the dataset

**Option A — Bundled copy** (if already in the repo):

```text
Smart Meter Electricity Consumption Dataset/smart_meter_data.csv
```

**Option B — Download from Kaggle:** [dataset page](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset). Place the CSV at the path above or at `data/raw/smart_meter_data.csv`.

**Option C — Kaggle API:**

```bash
pip install kaggle
kaggle datasets download -d ziya07/smart-meter-electricity-consumption-dataset -p data/raw --unzip
```

---

## 3. Smoke-check ingestion

```bash
python -m src.data.ingest_data
```

Expect: schema summary shape `(5000, 7)`, zero nulls, continuity check **PASS**.

Optional notebook: `notebooks/01_data_ingestion_and_schema_check.ipynb` (Colab/Kaggle path tips are in the notebook cells).

---

## 4. Run the end-to-end pipeline

```bash
python main.py --model naive
```

This runs ingest → features → anomaly detection → cleaning → time-ordered split → forecast → metrics → `data/processed/final_predictions.csv`.

Other models: `--model prophet`, `xgboost`, or `lstm`. Full flags and logs: [E2E Pipeline](e2e-pipeline.md).

---

## 5. Useful follow-ons

| Task | Command or link |
|------|-----------------|
| Explore patterns | `notebooks/02_exploratory_data_analysis.ipynb` · [EDA Insights](eda-insights.md) |
| Produce Phase 2 clean CSV for scripts | `python scripts/generate_clean_data.py` · [Clean Dataset](clean-data.md) |
| Compare all forecast models | `python scripts/compare_forecasts.py` · [Forecast Model Comparison](forecast-model-comparison.md) |
| Teaching walkthrough | `notebooks/04_forecasting_tutorial.ipynb` · [Forecasting Tutorial](forecasting-tutorial.md) |
| Preview docs site | `mkdocs serve` → http://127.0.0.1:8000 |

More Phase 2 scripts (tuning, notebooks): [Anomaly Detection](anomaly-detection.md). Individual model pages live under **Forecasting** in the site nav.

---

## Troubleshooting

### `FileNotFoundError: No CSV found`

Put the file at `data/raw/smart_meter_data.csv` or `Smart Meter Electricity Consumption Dataset/smart_meter_data.csv`.

### `ModuleNotFoundError: No module named 'pandas'`

Activate `.venv` and run `pip install -r requirements.txt`. In Jupyter, select the `.venv` kernel.

### Wrong `PROJECT_ROOT` in a notebook

Set `PROJECT_ROOT` to the absolute path of the repository root.

---

## Next reading

1. [Architecture](architecture.md) — how the repo is organized  
2. [E2E Pipeline](e2e-pipeline.md) — CLI details  
3. [Forecast Model Comparison](forecast-model-comparison.md) — results table  
4. [Glossary](glossary.md) — shared terms  

??? info "Technical deep dive"

    **Ingest:** `python -m src.data.ingest_data`  
    **E2E:** `python main.py --model naive` (optional `--save_clean_data`, `--output_path`)  
    **Clean artifact for offline scripts:** `python scripts/generate_clean_data.py`  
    **Ladder:** `python scripts/compare_forecasts.py`  
    **Docs:** `mkdocs serve`

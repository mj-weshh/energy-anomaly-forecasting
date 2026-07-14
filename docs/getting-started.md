# Getting Started

Install the project, load the dataset, and verify schema completeness.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 or later |
| Git | Any recent version |
| pip | Bundled with Python |

---

## 1. Clone and Set Up (Local)

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

## 2. Obtain the Dataset

### Option A — Use the bundled local copy

If the CSV is already in the repository under:

```
Smart Meter Electricity Consumption Dataset/smart_meter_data.csv
```

No additional download is required. The ingestion script will find it automatically.

### Option B — Download from Kaggle

1. Create a Kaggle account and accept the dataset terms.
2. Go to [Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset).
3. Download `smart_meter_data.csv`.
4. Place it in either:
   - `Smart Meter Electricity Consumption Dataset/smart_meter_data.csv`, or
   - `data/raw/smart_meter_data.csv` (recommended canonical location)

### Option C — Kaggle API

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (Linux/macOS) or C:\Users\<you>\.kaggle\ (Windows)

kaggle datasets download -d ziya07/smart-meter-electricity-consumption-dataset -p data/raw --unzip
```

---

## 3. Run Ingestion and Verification

```bash
python -m src.data.ingest_data
```

Expected output sections:

1. **Project root** and **Loaded CSV** paths
2. **SCHEMA SUMMARY** — shape `(5000, 7)`, dtypes, zero nulls
3. **TIME-SERIES CONTINUITY CHECK** — result `PASS`

---

## 4. Run the Ingestion Notebook

Open and execute all cells in:

```
notebooks/01_data_ingestion_and_schema_check.ipynb
```

### Local (default)

Cell 2 uses:

```python
PROJECT_ROOT = Path("..").resolve()
```

No changes needed if the notebook is opened from the `notebooks/` directory.

### Google Colab

Uncomment the Colab block in cell 2:

```python
from google.colab import drive
drive.mount("/content/drive")
PROJECT_ROOT = Path("/content/drive/MyDrive/energy-anomaly-forecasting")
```

Clone or copy the repo to Google Drive first, and ensure the CSV is present.

### Kaggle Notebooks

Uncomment the Kaggle block in cell 2:

```python
!pip install -q kaggle
!kaggle datasets download -d ziya07/smart-meter-electricity-consumption-dataset -p /tmp/data --unzip
PROJECT_ROOT = Path("/tmp/data")
```

Alternatively, add the dataset through the Kaggle notebook **Add Data** sidebar and point `PROJECT_ROOT` to `/kaggle/input/<dataset-slug>`.

---

## 5. Run the EDA Notebook

After ingestion passes, open and execute:

```
notebooks/02_exploratory_data_analysis.ipynb
```

This notebook profiles feature distributions, temporal load patterns, weather correlations, and the anomaly label baseline using `src/visualization/visualize.py`.

Documented findings and figures: [EDA Insights](eda-insights.md).

---

## 6. Regenerate EDA Documentation Assets

To refresh PNG figures embedded in the docs site:

```bash
python scripts/export_eda_assets.py
```

Output directory: `docs/assets/eda/`

---

## 7. Phase 2 Anomaly Detection Scripts

After feature engineering is in place, run the Phase 2 verification and clean-data scripts:

```bash
python scripts/verify_features.py
python scripts/test_isolation_forest.py
python scripts/tune_dbscan.py
python scripts/generate_clean_data.py
```

Expected outcomes:

- `verify_features.py` — engineered columns present; rolling warm-up NaNs as designed
- `test_isolation_forest.py` — IF baseline F1 ≈ 0.331 on 4,953 eval rows
- `tune_dbscan.py --legacy` — legacy coarse grid; enhanced mode uses scaled features
- `generate_clean_data.py` — writes `data/processed/clean_smart_meter_data.csv` (5000 × 15, 0 NaNs)

Research tuning (enhanced features, temporal 60/20/20 splits):

```bash
python scripts/tune_isolation_forest.py
python scripts/tune_dbscan.py
python scripts/tune_ensemble.py
python scripts/compare_anomaly_models.py
```

Expected: enhanced IF test F1 ≈ **0.46**; legacy IF fair test F1 ≈ **0.34**; ensemble union test F1 ≈ **0.40** (see `anomaly_config.py`).

Full results: [Anomaly Tuning Results](anomaly-tuning-results.md) · [Anomaly Detection — Research Tuning](anomaly-detection.md#research-tuning-enhanced-features--temporal-splits) · [Clean Dataset](clean-data.md)

---

## 8. Run the Anomaly Detection Notebook

Open and execute all cells in:

```
notebooks/03_anomaly_detection.ipynb
```

This CMU-Africa educational tutorial walks through load → features → IF/DBSCAN benchmark → interpolation, importing canonical modules from `src/` rather than reimplementing logic inline.

Documented workflow: [Anomaly Detection — Educational Notebook (Day 4)](anomaly-detection.md#educational-notebook-day-4) · [Clean Dataset](clean-data.md)

---

## 9. Build Documentation Site (Optional)

```bash
pip install mkdocs mkdocs-material
mkdocs serve    # preview at http://127.0.0.1:8000
mkdocs build    # output to site/
```

---

## Troubleshooting

### `FileNotFoundError: No CSV found under ...`

**Cause:** The ingestion script cannot locate `smart_meter_data.csv`.

**Fix:** Confirm the file exists at one of these paths:

```
data/raw/smart_meter_data.csv
Smart Meter Electricity Consumption Dataset/smart_meter_data.csv
```

### Wrong `PROJECT_ROOT` in the notebook

**Cause:** Notebook is running from an unexpected working directory.

**Fix:** Set `PROJECT_ROOT` to the absolute path of the repository root:

```python
PROJECT_ROOT = Path("/full/path/to/energy-anomaly-forecasting")
```

### `ModuleNotFoundError: No module named 'pandas'`

**Cause:** Virtual environment not activated or dependencies not installed.

**Fix:**

```bash
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### `ImportError` or slow pandas import in the notebook

**Cause:** Kernel using a different Python interpreter than the venv.

**Fix:** In VS Code or Jupyter, select the kernel associated with `.venv`.

---

## Next Steps

After ingestion and Phase 2 pass:

1. Review the [Data Schema](data-schema.md) reference
2. Confirm Week 1 results in the [Verification Report](verification-report.md)
3. Run `notebooks/02_exploratory_data_analysis.ipynb` and review [EDA Insights](eda-insights.md)
4. Run Phase 2 scripts (§7) and `notebooks/03_anomaly_detection.ipynb` (§8)
5. Generate the Phase 3 artifact: `python scripts/generate_clean_data.py`
6. Proceed to **Phase 3** — time-series forecasting on the clean dataset

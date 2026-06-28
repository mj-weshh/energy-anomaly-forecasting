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
git clone https://github.com/<your-org>/energy-anomaly-forecasting.git
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

## 7. Build Documentation Site (Optional)

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

After ingestion passes:

1. Review the [Data Schema](data-schema.md) reference
2. Confirm results in the [Verification Report](verification-report.md)
3. Run `notebooks/02_exploratory_data_analysis.ipynb` and review [EDA Insights](eda-insights.md)
4. Proceed to Phase 2 — unsupervised anomaly detection (planned)

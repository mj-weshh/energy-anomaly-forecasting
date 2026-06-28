# Architecture

Repository layout, data flow, and design decisions for Phase 1.

---

## System Overview

```mermaid
flowchart LR
    kaggle[KaggleDataset] --> rawCsv[smart_meter_data.csv]
    rawCsv --> ingest[src/data/ingest_data.py]
    ingest --> validatedDF[ValidatedDataFrame]
    validatedDF --> visualize[src/visualization/visualize.py]
    validatedDF --> eda[Phase1_EDA]
    validatedDF --> anomaly[Phase2_AnomalyDetection]
    validatedDF --> forecast[Phase3_Forecasting]
    visualize --> docAssets[docs/assets/eda]
```

The ingestion layer is the **single gate** between raw CSV files and all downstream work. Every notebook and script in later phases should import from `src.data.ingest_data` rather than reading CSVs directly.

---

## Repository Layout

```
energy-anomaly-forecasting/
├── data/
│   └── raw/                          # Canonical location for raw CSV (optional)
├── docs/                             # Project documentation (MkDocs source)
│   └── assets/                       # Screenshots and static assets
│       └── eda/                      # Phase 1 Week 2 EDA figures (PNG)
├── notebooks/
│   ├── 01_data_ingestion_and_schema_check.ipynb
│   └── 02_exploratory_data_analysis.ipynb
├── scripts/
│   ├── export_eda_assets.py          # Regenerate EDA doc figures
│   └── setup-git-hooks.ps1
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── ingest_data.py            # Canonical ingestion module
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py              # EDA plotting functions
├── Smart Meter Electricity Consumption Dataset/
│   └── smart_meter_data.csv          # Current raw data location
├── .gitignore
├── LICENSE
├── mkdocs.yml
├── README.md
└── requirements.txt
```

### Directory rationale

| Path | Purpose |
|------|---------|
| `src/` | Reusable, importable Python modules |
| `notebooks/` | Interactive workflows for exploration and reporting |
| `data/raw/` | Future canonical storage for raw files |
| `docs/` | Human-readable documentation source |
| `docs/assets/eda/` | Exported EDA plots for MkDocs |
| `scripts/` | CLI utilities (asset export, git hooks) |
| `Smart Meter Electricity Consumption Dataset/` | Legacy download location; supported by dynamic discovery |

---

## Ingestion Pipeline

```mermaid
flowchart TD
    start[main] --> findCsv[find_dataset_csv]
    findCsv --> load[load_smart_meter_data]
    load --> parseTimestamp[Parse Timestamp to datetime64]
    parseTimestamp --> sortRows[Sort chronologically]
    sortRows --> schema[print_schema_summary]
    schema --> continuity[check_time_continuity]
    continuity --> done[Report PASS or REVIEW]
```

### Module: `src/data/ingest_data.py`

| Function | Responsibility |
|----------|----------------|
| `get_project_root()` | Resolve repository root from module location |
| `find_dataset_csv(root)` | Dynamically locate CSV with fallback search paths |
| `load_smart_meter_data(csv_path)` | Read CSV, parse timestamps, sort rows |
| `print_schema_summary(df)` | Report shape, columns, dtypes, null counts |
| `check_time_continuity(df)` | Validate 30-minute cadence, gaps, duplicates |
| `main()` | CLI entry point orchestrating the full pipeline |

**CLI usage:**

```bash
python -m src.data.ingest_data
```

---

## Visualization Module

### Module: `src/visualization/visualize.py`

| Function | Responsibility |
|----------|----------------|
| `add_temporal_features(df)` | Derive hour, day-of-week, and weekend flags |
| `plot_feature_histograms(df)` | Numeric feature distributions with KDE |
| `plot_hourly_load_profile(df)` | Daily load profile boxplot by hour |
| `plot_weekly_load_profile(df)` | Weekly seasonality boxplot by day |
| `plot_correlation_heatmap(df)` | Pearson correlation matrix heatmap |
| `plot_anomaly_label_distribution(df)` | Bar chart of Normal vs Abnormal counts |
| `plot_consumption_timeseries(df)` | Time series with rolling mean overlay |

**Asset export:**

```bash
python scripts/export_eda_assets.py
```

Writes PNGs to `docs/assets/eda/` for embedding in [EDA Insights](eda-insights.md).

---

## Design Decisions

### Canonical source vs. notebook duplication

| Artifact | Role |
|----------|------|
| `src/data/ingest_data.py` | **Canonical source** — import in scripts, tests, and future modules |
| `notebooks/01_data_ingestion_and_schema_check.ipynb` | **Portable copy** — inline functions for Colab/Kaggle where `src/` may not be on `PYTHONPATH` |
| `src/visualization/visualize.py` | **Canonical EDA plots** — shared by notebook and export script |

When ingestion or visualization logic changes, update the Python module first, then sync the notebook.

### Dynamic CSV discovery

Hard-coding absolute paths breaks portability across local, Colab, and Kaggle environments. `find_dataset_csv()` searches multiple candidate locations in priority order so the same code works regardless of where the user placed the download.

### Phase gate: schema before EDA

No exploratory analysis or modeling runs until:

- Schema completeness is verified (7 columns, zero nulls)
- Time-series continuity passes (30-minute intervals, no gaps)

This prevents silent data quality issues from propagating into Phase 2 and Phase 3.

### Documentation figures from scripts

EDA plots in MkDocs are generated by `scripts/export_eda_assets.py` rather than extracted from notebook outputs. This keeps figures reproducible and avoids committing large base64 blobs in `.ipynb` files.

---

## Phase Roadmap

| Phase | Deliverables | Key modules |
|-------|-------------|-------------|
| **1 — Setup & EDA** | Ingestion, schema validation, EDA, documentation | `src/data/ingest_data.py`, `src/visualization/visualize.py` |
| **2 — Anomaly Detection** | Isolation Forest, DBSCAN, evaluation | `src/models/` (planned) |
| **3 — Forecasting** | XGBoost, LSTM, evaluation | `src/models/` (planned) |

---

## Technology Stack (Phase 1)

| Component | Library | Version constraint |
|-----------|---------|-------------------|
| Data manipulation | pandas | >= 2.0.0 |
| Numerical computing | numpy | >= 1.24.0 |
| Environment config | python-dotenv | >= 1.0.0 |
| Notebooks | jupyter, ipykernel | >= 1.0.0, >= 6.0.0 |
| Visualization | matplotlib, seaborn | >= 3.7.0, >= 0.13.0 |
| Statistics | scipy, statsmodels | >= 1.11.0, >= 0.14.0 |
| Documentation | mkdocs, mkdocs-material | >= 1.6.0, >= 9.5.0 |

ML libraries (scikit-learn, xgboost, tensorflow/pytorch) will be added in Phase 2 and Phase 3.

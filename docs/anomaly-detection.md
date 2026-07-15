# Anomaly Detection — Phase 2, Week 4

Working notes on the unsupervised anomaly detection engine. Week 3 gave us a 15-column feature matrix; Week 4 wires that into two detectors with proper imbalance-aware evaluation and a unified routing API.

!!! success "Executive summary"

    - **What we built:** Two automatic ways to spot unusual electricity readings — Isolation Forest (primary) and DBSCAN (comparison).
    - **Production today:** Default cleaning uses Isolation Forest with standard settings; catches about **one third** of benchmark problems (F1 0.331) but keeps the pipeline simple and stable.
    - **Research path:** Tuned models on future-held-out data score **0.460 F1** — materially better, but not yet wired into the default clean file.
    - **False alarms:** Early-morning hours (00–01) drive most legacy false positives; see [Anomaly Tuning Results — Error analysis](anomaly-tuning-results.md#legacy-if-error-analysis-hourly-fp).
    - **Terms:** [Glossary](glossary.md) — F1, contamination, temporal split.

**Status:** Week 4 complete (Days 1–4) — detectors, comparison, clean dataset pipeline, and educational notebook  
**Modules:** `src/models/evaluate_models.py`, `src/models/train_anomaly_models.py`  
**Builds on:** [Phase 2 Strategy](phase2-strategy.md), [Feature Engineering](feature-engineering.md)

---

## Why Isolation Forest First

The strategy doc describes what we need: a detector that flags readings that don't fit the local pattern, not a dumb threshold on raw consumption. Isolation Forest fits that story — it scores **multivariate weirdness** across all 12 engineered features (consumption, weather, temporal context, rolling memory) without needing balanced labels.

I started here before DBSCAN because IF is fast, interpretable at the pipeline level, and gives us a baseline F1 to beat. DBSCAN is the second opinion with different geometry assumptions.

---

## What's Implemented

### Evaluation — `evaluate_anomaly_model(y_true, y_pred)`

Imbalance-aware metrics for the benchmark comparison. Treats class **1** (Abnormal) as the positive class — no accuracy metric, because a model that always predicts Normal would hit 95%.

Returns precision, recall, F1, and a 2×2 confusion matrix.

### Training — `train_isolation_forest(df, contamination=0.05)`

Unsupervised fit on the engineered feature matrix. Critical design choices:

| Rule | Why |
|------|-----|
| **`Anomaly_Label` never used for training** | Labels are a benchmark only; the model must not see them during fit |
| **`prepare_feature_matrix` drops Timestamp + label + NaN rows** | 47 rolling warm-up rows removed; **4953 rows** remain for training |
| **12 numeric features** | Raw consumption + weather + temporal + rolling columns |
| **`contamination=0.05`** | Matches the ~5% Abnormal rate in the dataset; unsupervised prior, not supervised tuning |
| **Predictions mapped to 0/1** | sklearn returns `-1` (outlier) / `1` (inlier); we map to **1 = Abnormal, 0 = Normal** for evaluation |

### Training — `train_dbscan(df, eps=0.5, min_samples=5)`

Density-based second detector on the same feature matrix. Same label-exclusion rules as IF:

| Rule | Why |
|------|-----|
| **`Anomaly_Label` never used for training** | Benchmark only — same unsupervised contract as IF |
| **Same 4953 × 12 matrix** | Reuses `prepare_feature_matrix`; no duplicate prep logic |
| **Noise = anomaly** | DBSCAN returns `-1` for sparse points; cluster members (`>= 0`) are normal baseline density |
| **Predictions mapped to 0/1** | `-1` → **1 = Abnormal**, `>= 0` → **0 = Normal** |

DBSCAN is sensitive to `eps` and `min_samples` on multivariate 0–1 features — Day 2 includes a coarse grid search to find the best F1 on our benchmark.

### Unified router — `detect_anomalies(df, model_type='isolation_forest', **kwargs)`

Single entry point for notebooks and downstream evaluation:

- `model_type="isolation_forest"` → `train_isolation_forest(df, **kwargs)`
- `model_type="dbscan"` → `train_dbscan(df, **kwargs)`

Returns `(fitted_model, predictions)` with the same 0/1 encoding as the individual trainers.

### Pipeline helper — `build_all_features(df)`

One-call wrapper in `src/features/build_features.py`: temporal features, then rolling metrics.

### Verification scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_isolation_forest.py` | End-to-end IF baseline + benchmark evaluation |
| `scripts/tune_dbscan.py` | 12-combo grid search over `eps` × `min_samples`; prints best F1 |

Both scripts: load → features → extract benchmark labels (aligned to `dropna` rows) → train unsupervised → evaluate.

---

## Baseline Results — Isolation Forest (Day 1)

Output from `python scripts/test_isolation_forest.py` on the real dataset:

```
Evaluation rows: 4953

Isolation Forest baseline metrics (Abnormal = positive class):
  Precision: 0.331
  Recall:    0.331
  F1:        0.331

Confusion matrix [[TN, FP], [FN, TP]]:
  TN=4539  FP= 166
  FN= 166  TP=  82
```

| Metric | Value | Notes |
|--------|-------|-------|
| Evaluation rows | 4,953 | 5,000 minus 47 rolling warm-up rows |
| Precision | 0.331 | Of predicted anomalies, ~33% match the benchmark |
| Recall | 0.331 | Of 248 benchmark Abnormal rows in eval set, ~82 caught |
| F1 | 0.331 | Harmonic mean; baseline, not tuned |
| Predicted anomalies | 248 | Close to the ~5% benchmark rate |

Honest read: IF catches roughly **one third** of benchmark anomalies at default contamination. That's our bar for DBSCAN and future tuning.

---

## Baseline Results — DBSCAN (Day 2)

Output from `python scripts/tune_dbscan.py` — grid over `eps` ∈ {0.1, 0.3, 0.5, 0.7} and `min_samples` ∈ {5, 10, 20}:

```
DBSCAN grid search — best F1 (Abnormal = positive):
  eps=0.5  min_samples=5
  Precision: 0.084
  Recall:    0.246
  F1:        0.125
  Predicted anomalies: 726 / 4953

Confusion matrix [[TN, FP], [FN, TP]]:
  TN=4040  FP= 665
  FN= 187  TP=  61
```

| Metric | Value | Notes |
|--------|-------|-------|
| Best config | `eps=0.5`, `min_samples=5` | Winner on this coarse 12-combo grid |
| F1 | 0.125 | Below IF baseline (0.331) |
| Precision | 0.084 | Many false positives — flags ~15% of rows as anomalous |
| Recall | 0.246 | Catches ~61 of 248 benchmark Abnormal rows |
| Predicted anomalies | 726 | Over-predicts relative to the 5% benchmark |

Honest read: on this grid, DBSCAN **underperforms** Isolation Forest. Most combinations either flag nearly everything (F1 ≈ 0.095) or miss too many true anomalies. The best combo still over-flags. Finer `eps` search or feature scaling may help, but IF is the stronger baseline on this dataset today.

---

## Model Comparison

Same 4,953 evaluation rows, labels excluded from all training:

| Model | Config | F1 | Precision | Recall | Pred. anomalies |
|-------|--------|-----|-----------|--------|-----------------|
| **Isolation Forest** | `contamination=0.05` | **0.331** | **0.331** | **0.331** | 248 |
| **DBSCAN (grid best)** | `eps=0.5`, `min_samples=5` | 0.125 | 0.084 | 0.246 | 726 |

**Current pick:** Isolation Forest on F1 for production cleaning. Ensemble is implemented for research (`train_ensemble`, `scripts/tune_ensemble.py`); union test F1 = **0.400** — below enhanced IF alone (0.460). See [Anomaly Tuning Results](anomaly-tuning-results.md).

---

## Educational Notebook (Day 4)

CMU-Africa deliverable: an interactive tutorial that walks through the full Phase 2 workflow — unsupervised detection, benchmark evaluation, and consumption interpolation for Phase 3 forecasting.

**Notebook:** [`notebooks/03_anomaly_detection.ipynb`](../notebooks/03_anomaly_detection.ipynb)

| Section | Topic |
|---------|-------|
| 1 | Setup and data loading via `load_smart_meter_data` |
| 2 | Feature engineering via `build_all_features` |
| 3 | Isolation Forest and DBSCAN explained |
| 4 | Train both models with `detect_anomalies`; benchmark with `evaluate_anomaly_model` |
| 5 | Clean series with `interpolate_anomalies` (IF predictions); before/after plot |

The notebook imports canonical `src/` modules — same APIs as the verification scripts, not inline reimplementation.

**Run locally:**

```bash
jupyter notebook notebooks/03_anomaly_detection.ipynb
# or headless:
jupyter nbconvert --execute notebooks/03_anomaly_detection.ipynb
```

**Verified outputs on the canonical dataset:**

| Check | Value |
|-------|-------|
| Evaluation rows | 4,953 (47 rolling warm-up rows excluded) |
| Isolation Forest F1 | 0.331 |
| DBSCAN best F1 | 0.125 (`eps=0.5`, `min_samples=5`) |
| Imputed consumption values | ~248 (IF at `contamination=0.05`) |
| Shape after cleaning | `(5000, 15)` — row count preserved |

Interpolation details: [Clean Dataset](clean-data.md).

---

## Research Tuning (Enhanced Features + Temporal Splits)

Production cleaning still uses **legacy** 15-column features and default IF (`contamination=0.05`). Research scripts opt into enhanced features (`build_enhanced_anomaly_features`, 21 columns), train-only scaling, chronological 60/20/20 splits, and hyperparameter grids.

**Status:** Tuning complete — full report: [Anomaly Tuning Results](anomaly-tuning-results.md)

**Modules:** `src/models/anomaly_preprocessing.py`, `src/models/tuning_utils.py`, `src/models/anomaly_config.py`

| Script | Purpose |
|--------|---------|
| `scripts/tune_isolation_forest.py` | IF grid + score-threshold tuning on validation |
| `scripts/tune_isolation_forest.py --drop-weather` | Weather ablation — drop Temperature, Humidity, Wind_Speed |
| `scripts/tune_dbscan.py` | DBSCAN grid (enhanced scaled or `--legacy`) |
| `scripts/tune_ensemble.py` | IF ∪ DBSCAN union/intersection/weighted strategies |
| `scripts/compare_anomaly_models.py` | Legacy vs enhanced dashboard + fair test-split comparison |
| `scripts/analyze_detection_errors.py` | Legacy IF hourly false-positive breakdown on test split |
| `scripts/tune_isolation_forest_by_segment.py` | Per-hour/weekend enhanced IF test F1 (global model) |
| `scripts/compare_clean_artifacts.py` | Diff legacy vs research clean CSV profiles |

### Tuned results (summary)

| Model | Test F1 | Notes |
|-------|---------|-------|
| **Legacy IF** (full dataset) | **0.331** | production baseline |
| **Legacy IF** (fair test, production params) | **0.340** | same window as enhanced |
| **Legacy IF** (fair test, val threshold) | **0.389** | same protocol as enhanced, no hyperparam grid |
| **Enhanced IF** | **0.460** | +0.071 F1 (~18% relative) vs val-threshold legacy (0.389) |
| **Enhanced DBSCAN** | **0.297** | below IF |
| **Ensemble (union)** | **0.400** | below enhanced IF alone |

See [Anomaly Tuning Results](anomaly-tuning-results.md) for methodology, confusion matrices, search spaces, and fair head-to-head tables.

Configs are stored in `src/models/anomaly_config.py`. Production `generate_clean_dataset` uses the **default `legacy` profile only**; research profiles are opt-in via `--profile` — see [Clean Dataset — Research profiles](clean-data.md#research-profiles).

---

## How to Reproduce

```bash
pip install -r requirements.txt
python scripts/test_isolation_forest.py
python scripts/tune_dbscan.py --legacy
python scripts/tune_isolation_forest.py
python scripts/tune_isolation_forest.py --drop-weather
python scripts/tune_dbscan.py
python scripts/tune_ensemble.py
python scripts/compare_anomaly_models.py
python scripts/analyze_detection_errors.py
python scripts/tune_isolation_forest_by_segment.py
```

Python API:

```python
from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import (
    detect_anomalies,
    train_dbscan,
    train_isolation_forest,
)

df = build_all_features(load_smart_meter_data(find_dataset_csv(get_project_root())))

# Individual trainers
if_model, if_preds = train_isolation_forest(df)
db_model, db_preds = train_dbscan(df, eps=0.5, min_samples=5)

# Unified router
_, preds = detect_anomalies(df, model_type="isolation_forest")
_, preds = detect_anomalies(df, model_type="dbscan", eps=0.5, min_samples=5)
```

??? info "Technical deep dive"

    **Modules:** `src/models/train_anomaly_models.py`, `evaluate_models.py`, `anomaly_preprocessing.py`, `tuning_utils.py`, `anomaly_config.py`.

    **Eval rows:** 4,953 (5,000 minus 47 rolling warm-up NaNs). Labels aligned via `prepare_feature_matrix` index.

    **Fair test F1 (991 rows):** legacy 0.340 / 0.389 (val threshold) / enhanced 0.460. See [Anomaly Tuning Results](anomaly-tuning-results.md).

    **Clean profiles:** `legacy` (default), `legacy_threshold`, `enhanced` — `scripts/generate_clean_data.py --profile`.

---

## What's Next

- **Phase 3 forecasting** — train on the clean artifact; see [Clean Dataset](clean-data.md)
- Optional: adopt tuned IF config for production cleaning (currently legacy by design)

---

## References

- [Phase 2 Strategy](phase2-strategy.md) — why context-aware detection and imbalance-aware metrics matter
- [Feature Engineering](feature-engineering.md) — the 12 features fed into the model
- [Clean Dataset](clean-data.md) — Day 3 imputation pipeline for Phase 3
- [Architecture](architecture.md) — where `src/models/` sits in the repo
- [Anomaly Tuning Results](anomaly-tuning-results.md) — enhanced features, temporal splits, fair head-to-head comparison

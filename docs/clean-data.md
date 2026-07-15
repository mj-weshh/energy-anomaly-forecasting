# Clean Dataset — Phase 2, Week 4 Day 3

Working notes on the clean-data pipeline. Week 4 Days 1–2 gave us anomaly detectors; Day 3 turns those predictions into a **continuity-safe** dataset for Phase 3 forecasting.

**Status:** Week 4 Day 3–4 complete — clean dataset pipeline, artifact generation, and notebook Section 5 walkthrough  
**Modules:** `src/data/clean_data.py`, `src/pipelines/clean_dataset.py`, `scripts/generate_clean_data.py`  
**Builds on:** [Anomaly Detection](anomaly-detection.md), [Feature Engineering](feature-engineering.md)

---

## Why Impute, Not Drop

Phase 3 forecasters (ARIMA, LSTM, XGBoost) assume a **continuous** 30-minute timeline. If we `.drop()` rows flagged as anomalous, we create gaps — irregular intervals that break time-series models and rolling features.

The fix: treat bad consumption readings as missing values, then **time-interpolate** them. Row count stays at **5000**; only `Electricity_Consumed` values change at flagged intervals.

---

## What's Implemented

### Masking and interpolation — `interpolate_anomalies(df, predictions)`

| Step | Behavior |
|------|----------|
| **Mask** | Where `predictions == 1` (Abnormal), set `Electricity_Consumed` to `NaN` |
| **Align** | Predictions from `detect_anomalies` may be length 4953; aligned via `prepare_feature_matrix` index — warm-up rows untouched |
| **Interpolate** | `interpolate(method="time")` on a temporary DatetimeIndex |
| **Preserve** | All columns and 5000 rows returned; input never mutated |

Only consumption gaps created by masking are filled — other columns are not interpolated.

### `generate_clean_dataset(input_path, output_path)`

Defined in `src/pipelines/clean_dataset.py` (re-exported from `src.data.clean_data` for compatibility). Chains:

1. `load_smart_meter_data` — raw CSV
2. Feature engineering (profile-dependent)
3. Anomaly detection (profile-dependent)
4. `interpolate_anomalies` — mask + impute
5. Write CSV to `output_path`

## Research profiles (`--profile`)

The default pipeline is **unchanged**. Optional profiles write separate research artifacts under `data/processed/`:

| Profile | Features | Detection | Default output |
|---------|----------|-----------|----------------|
| **`legacy`** (default) | `build_all_features` (15 cols) | IF defaults (`contamination=0.05`) on all eval rows | `clean_smart_meter_data.csv` |
| **`legacy_threshold`** | `build_all_features` | Train 60%, val threshold 20%, score all eval rows (fair eval F1 ≈ 0.389) | `clean_smart_meter_data_legacy_threshold.csv` |
| **`enhanced`** | `build_enhanced_anomaly_features` (21 cols) | `BEST_IF_CONFIG` + score threshold on all eval rows | `clean_smart_meter_data_enhanced.csv` |

```bash
python scripts/generate_clean_data.py                          # legacy (unchanged)
python scripts/generate_clean_data.py --profile legacy_threshold
python scripts/generate_clean_data.py --profile enhanced
python scripts/compare_clean_artifacts.py
```

Research profiles are **not** used for production Phase 3 baseline until artifact diffs are reviewed. See [Anomaly Tuning Results](anomaly-tuning-results.md).

---

## Output Artifact

From `python scripts/generate_clean_data.py` on the canonical dataset:

| Property | Value |
|----------|-------|
| Path | `data/processed/clean_smart_meter_data.csv` |
| Shape | **5000 × 15** (7 original + 8 engineered columns) |
| `Electricity_Consumed` NaNs | **0** after interpolation |
| Rows dropped | **0** — continuity preserved |
| Anomalies imputed | ~**248** intervals (IF at `contamination=0.05`) |

### Why Isolation Forest for cleaning

IF leads on benchmark F1 (0.331 vs DBSCAN best 0.125) and predicts closer to the ~5% anomaly rate (~248 flagged vs ~726 for DBSCAN defaults). Better fit for a conservative clean pass before forecasting.

Research tuning reached enhanced IF test F1 **0.460** on a held-out window; that config is **not** used for this artifact. See [Anomaly Tuning Results](anomaly-tuning-results.md).

---

## How to Reproduce

```bash
pip install -r requirements.txt
python scripts/generate_clean_data.py
```

Expected output:

```
Loaded: .../smart_meter_data.csv
Wrote:  .../data/processed/clean_smart_meter_data.csv
Shape:  (5000, 15)
Electricity_Consumed NaNs: 0
```

Python API:

```python
from src.data.clean_data import generate_clean_dataset, interpolate_anomalies
from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import detect_anomalies

df = build_all_features(load_smart_meter_data(find_dataset_csv(get_project_root())))
_, preds = detect_anomalies(df, model_type="isolation_forest")
clean_df = interpolate_anomalies(df, preds)

# Or one call:
generate_clean_dataset(
    str(find_dataset_csv(get_project_root())),
    "data/processed/clean_smart_meter_data.csv",
)
```

---

## What's Next

- **Phase 3** — train forecasting models on `clean_smart_meter_data.csv`
- **Notebook walkthrough** — Section 5 of [`notebooks/03_anomaly_detection.ipynb`](../notebooks/03_anomaly_detection.ipynb) demonstrates masking, interpolation, and a before/after consumption plot; see [Educational Notebook (Day 4)](anomaly-detection.md#educational-notebook-day-4)

---

## References

- [Anomaly Detection](anomaly-detection.md) — Isolation Forest baseline used for cleaning
- [Feature Engineering](feature-engineering.md) — features included in the output CSV
- [Architecture](architecture.md) — where `src/data/clean_data.py` sits in the repo
- [Anomaly Tuning Results](anomaly-tuning-results.md) — research IF configs (not adopted for cleaning)

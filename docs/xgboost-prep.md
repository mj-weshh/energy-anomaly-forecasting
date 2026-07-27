# XGBoost Prep — Phase 3, Week 7 (Day 1)

Working notes for preparing the first advanced ML forecaster: convert the cleaned half-hour timeline into a **supervised tabular** frame with lag predictors so gradient-boosted trees can learn from past consumption.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Why lags:** XGBoost does not read clock time — each row must carry past consumption as explicit columns beside the target at time **t**.
- **What we add:** Three lag features — **t-1** (30 min), **t-2** (1 hour), **t-48** (24 hours at 30-min resolution).
- **Row trim:** Shifting drops the first **48** rows on a continuous clean series (**5000 -> 4952** rows) so every remaining row has complete predictors.
- **Verify:** `python scripts/verify_xgboost_prep.py` prints shape, columns, and a sample to confirm alignment.
- **Terms:** [Glossary](glossary.md) — supervised lag features, tabular forecasting frame.

</div>

**Status:** Week 7 Day 1 complete — supervised lag function and verification script  
**Module:** `src/features/build_features.py` — `create_supervised_lags`  
**Script:** `scripts/verify_xgboost_prep.py`  
**Builds on:** [Forecasting Baseline](forecasting-baseline.md), [Feature Engineering](feature-engineering.md), [Phase 3 Strategy](phase3-strategy.md)

---

## Why Tabular Lags

Statistical models like Prophet map timestamps natively. Tree models like **XGBoost** treat each row as a flat feature vector — they need **past values on the same row** as predictors for the target at time **t**.

| Model family | Time handling |
|--------------|---------------|
| Prophet / ARIMA | Native datetime / seasonality |
| XGBoost | Lag columns + optional temporal features from Phase 2 |

Phase 1 EDA showed strong time-of-day structure. Lag **48** captures the same 24-hour persistence as the naive seasonal baseline; lags **1** and **2** add short-term momentum.

---

## `create_supervised_lags`

Module: `src/features/build_features.py`

```python
create_supervised_lags(df, target_col="Electricity_Consumed") -> pd.DataFrame
```

| Step | Behavior |
|------|----------|
| **Validate** | Requires `target_col` in `df`; raises `KeyError` if missing |
| **Sort** | Chronological sort by `Timestamp` when present |
| **Shift** | Adds `{target_col}_lag_1`, `_lag_2`, `_lag_48` via `Series.shift` |
| **Drop NaNs** | Removes rows with incomplete lag history; resets index |
| **Copy semantics** | Never mutates the caller's DataFrame |

### Lag column reference

| Column | Shift | Meaning |
|--------|-------|---------|
| `Electricity_Consumed_lag_1` | 1 | Consumption 30 minutes ago |
| `Electricity_Consumed_lag_2` | 2 | Consumption 1 hour ago |
| `Electricity_Consumed_lag_48` | 48 | Consumption 24 hours ago |

### Why drop the first 48 rows?

XGBoost can tolerate NaNs, but we drop incomplete rows so **all forecast models evaluate on the same timeline** after feature warm-up. On the default clean artifact (`5000` continuous rows), output shape is **4952 x 18** (original clean columns plus three lags).

Usage:

```python
import pandas as pd
from src.features.build_features import create_supervised_lags

df = pd.read_csv("data/processed/clean_smart_meter_data.csv", parse_dates=["Timestamp"])
tabular = create_supervised_lags(df)
```

---

## Verify Tabular Prep

```bash
python scripts/verify_xgboost_prep.py
```

Workflow: load `data/processed/clean_smart_meter_data.csv` -> `create_supervised_lags` -> print input/output shape, column list, and `.head()` of target + lag columns.

### Expected output (local)

| Check | Value |
|-------|-------|
| Input shape | **5000 x 15** |
| Output shape | **4952 x 18** |
| Rows dropped | **48** |
| Lag columns | `_lag_1`, `_lag_2`, `_lag_48` present |

Inspect the sample rows: `Electricity_Consumed_lag_1` at row **t** should equal `Electricity_Consumed` from the prior interval in the sorted series.

---

## What's Next

Per [Phase 3 Strategy](phase3-strategy.md):

1. XGBoost trainer on the chronological **70/15/15** split
2. Score with `evaluate_forecast` on the held-out test window
3. Compare MAE / RMSE against the naive floor and Prophet baseline
4. Optionally add Phase 2 temporal columns (`hour`, `day_of_week`, rolling stats) as extra predictors

XGBoost training and evaluation scripts are **not yet implemented** — this day stops at verified tabular prep.

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Module map:** `create_supervised_lags` lives alongside Phase 2 helpers in `build_features.py`. Day 1 verify uses the clean CSV directly; full XGBoost training will chain `time_series_split` + lags + trainer.

**Relationship to Phase 2 features:** The clean artifact already includes temporal and rolling columns from production cleaning. Lag prep adds consumption history columns; temporal features can be reused as XGBoost inputs in a later step.

**Commands:**

```bash
python scripts/generate_clean_data.py          # if artifact missing
python scripts/verify_xgboost_prep.py
python scripts/verify_features.py              # Phase 2 feature sanity check
```

**Fair comparison note:** Apply `create_supervised_lags` **after** chronological split in training code so validation/test rows only use past values from allowed history — the verify script runs on the full clean series for shape inspection only.

</details>

---

## References

- [Phase 3 Strategy](phase3-strategy.md) — model ladder and XGBoost plan
- [Forecasting Baseline](forecasting-baseline.md) — chronological split and naive floor
- [Feature Engineering](feature-engineering.md) — Phase 2 temporal and rolling features
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — supervised lag features, tabular forecasting frame
- [Getting Started](getting-started.md) — install and Phase 3 commands

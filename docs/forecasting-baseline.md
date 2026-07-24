# Forecasting Baseline — Phase 3, Week 6 (Day 1–2)

Working notes for the Phase 3 foundation: prove the Phase 2 clean artifact is forecast-ready, split chronologically (no shuffle), define MAE / RMSE / MAPE, and lock a **naive seasonal** floor that advanced models must beat.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Gate first:** We refuse to train forecasts on broken timelines — 5,000 rows, zero consumption gaps, perfect 30-minute spacing.
- **No peeking at the future:** Train / validation / test are cut **in time order** (70% / 15% / 15%), never randomly shuffled.
- **Floor to beat:** “Same time yesterday” (48 half-hour steps = 24 hours). If a complex model cannot beat this on the test window, it is not useful.
- **Example test floor (reproducible):** MAE ≈ **0.171**, RMSE ≈ **0.214** on normalized consumption — run `evaluate_naive_baseline.py` to refresh.
- **Terms:** [Glossary](glossary.md) — MAE, RMSE, MAPE, seasonal naive, forecast chronological split.

</div>

**Status:** Week 6 Day 1–2 complete — clean-state audit, chronological split, metrics module, naive baseline scored  
**Modules:** `src/data/make_forecast_dataset.py`, `src/models/evaluate_forecast.py`, `src/models/train_forecast_models.py`  
**Scripts:** `scripts/verify_phase2_state.py`, `scripts/evaluate_naive_baseline.py`  
**Builds on:** [Clean Dataset](clean-data.md), [Phase 3 Strategy](phase3-strategy.md)

---

## Step 0 — Verify Phase 2 Clean State

Forecasting must not retrain Isolation Forest by default. The gate **loads** `data/processed/clean_smart_meter_data.csv` only.

```bash
python scripts/generate_clean_data.py   # if the artifact is missing
python scripts/verify_phase2_state.py
```

| Check | Pass criterion |
|-------|----------------|
| File exists | Path under `data/processed/` |
| Shape | Exactly **5,000** rows |
| Consumption | **0** NaNs in `Electricity_Consumed` |
| Continuity | Perfect **30-minute** grid (no gaps / duplicates / irregular deltas) |

On success the script prints `PASS — Phase 2 clean dataset is forecast-ready` and exits 0.

---

## Chronological Train / Validation / Test Split

Random `train_test_split` would leak future intervals into training. Phase 3 uses position slices after sorting by `Timestamp`:

| Split | Fraction | Role on 5,000-row clean CSV |
|-------|----------|------------------------------|
| Train | 70% | ~3,500 rows — learn patterns |
| Validation | 15% | ~750 rows — tuning / early stopping (unused by naive) |
| Test | 15% | ~750 rows — reported business metrics |

API: `time_series_split(df, train_pct=0.7, val_pct=0.15)` in `src/data/make_forecast_dataset.py`.

Boundary check (prints start/end dates; requires train end + 30 min = val start, etc.):

```bash
python -m src.data.make_forecast_dataset
```

This is **not** the Phase 2 anomaly research split (60/20/20 on eval rows). See [Glossary](glossary.md).

---

## Evaluation Metrics

Module: `src/models/evaluate_forecast.py` (separate from Phase 2 `evaluate_models.py`).

| Metric | Meaning | Notes |
|--------|---------|-------|
| **MAE** | Average absolute miss | Primary management-friendly score |
| **RMSE** | Penalizes large misses | Uses `sklearn.metrics.root_mean_squared_error` (sklearn 1.6+; `squared=` removed from MSE) |
| **MAPE** | Relative error in **percent** | Denominator floored with `epsilon=1e-8`; can explode when true consumption is near zero on this normalized scale — treat MAE/RMSE as the headline floor |

Wrapper: `evaluate_forecast(y_true, y_pred) → {"mae", "rmse", "mape"}`.

---

## Naive Seasonal Baseline

Module: `src/models/train_forecast_models.py`

```text
naive_seasonal_forecast(train_series, test_series, seasonal_periods=48)
```

At 30-minute resolution, **48 steps = 24 hours**. Each test prediction is the observed value exactly 48 steps earlier on the chronological series `train || test_true` (seasonal persistence). Lag lookup uses **observed** history — not recursive predicted values.

Why it matters: Phase 1 EDA showed strong time-of-day structure. Advanced models (Prophet/ARIMA, XGBoost, LSTM) must beat this floor on the **same test window**.

---

## Score the Floor

```bash
python scripts/evaluate_naive_baseline.py
```

Workflow: load clean CSV → `time_series_split` → `naive_seasonal_forecast` on `Electricity_Consumed` → `evaluate_forecast` on the test set → print MAE / RMSE / MAPE.

### Example run (local, after sklearn RMSE fix)

| Metric | Value |
|--------|-------|
| MAE | **0.171150** |
| RMSE | **0.214034** |
| MAPE | Unstable on near-zero true values (~10⁷ % scale) — use MAE/RMSE for comparisons |

Re-run the script after regenerating the clean artifact; numbers may shift slightly if cleaning or data path changes.

---

## What's Next

Per [Phase 3 Strategy](phase3-strategy.md):

1. Statistical baselines (Prophet / Auto-ARIMA)
2. XGBoost with lag + temporal features
3. LSTM sliding windows
4. Research write-up and tutorial notebook (`forecasting-research.md`, `04_forecasting_tutorial.ipynb`)

Each model must be scored with the same chronological cut and `evaluate_forecast` helpers.

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Scripts:** `verify_phase2_state.py` (CSV-only gate), `evaluate_naive_baseline.py` (end-to-end floor).

**APIs:** `time_series_split`, `naive_seasonal_forecast`, `evaluate_forecast` / `mean_absolute_error_forecast` / `root_mean_squared_error_forecast` / `mean_absolute_percentage_error_forecast`.

**Commands:**

```bash
python scripts/generate_clean_data.py
python scripts/verify_phase2_state.py
python -m src.data.make_forecast_dataset
python scripts/evaluate_naive_baseline.py
```

**Modularity:** Baseline path does not call `detect_anomalies` or regenerate the clean file unless you run `generate_clean_data.py` explicitly.

</details>

---

## References

- [Phase 3 Strategy](phase3-strategy.md) — full forecasting ladder and evaluation protocol
- [Clean Dataset](clean-data.md) — Phase 2 imputation artifact
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — MAE, RMSE, MAPE, seasonal naive, forecast split
- [Getting Started](getting-started.md) — install and Phase 3 commands

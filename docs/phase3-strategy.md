# Phase 3 Strategy — Time-Series Forecasting

Planning notes for the final technical phase: forecasting. Phase 1 (ingestion / EDA) and Phase 2 (anomaly detection and cleaning) are complete. We now have a clean, continuous dataset where historical anomalies have been masked and interpolated.

!!! success "Executive summary"

    - **Goal:** Predict future electricity use from the cleaned half-hour timeline — so demand forecasting sits on trustworthy history, not gaps or deleted rows.
    - **Starting point:** Default Phase 2 clean file (`clean_smart_meter_data.csv`) — continuous, ~248 repaired intervals, production recipe unchanged.
    - **Golden rule:** Split data **in time order** (70% train / 15% validation / 15% test). Never shuffle — that would leak the future into the past.
    - **Model ladder:** Beat a simple “same time yesterday” baseline before trusting Prophet/ARIMA, then XGBoost, then LSTM.
    - **Shipped:** Naive floor, Prophet, XGBoost, LSTM, unified comparison, E2E CLI, tutorial, and research write-up — see linked pages below.
    - **Terms:** [Glossary](glossary.md) — imputation, temporal split; forecasting metrics (MAE / RMSE / MAPE).

**Status:** Phase 3 complete — model ladder, E2E CLI, tutorial, and research write-up shipped  

**Builds on:** [Clean Dataset](clean-data.md), [Anomaly Detection](anomaly-detection.md), [Feature Engineering](feature-engineering.md), [Architecture](architecture.md), [Forecasting Baseline](forecasting-baseline.md)

---

## What Phase 2 Handed Off

Phase 2 ends with a continuity-safe artifact ready for forecasting:

| Property | Value |
|----------|-------|
| Path | `data/processed/clean_smart_meter_data.csv` |
| Generator | `generate_clean_dataset()` / `scripts/generate_clean_data.py` (default **`legacy`** profile) |
| Shape | **5,000 × 15** (7 original + 8 engineered columns) |
| `Electricity_Consumed` NaNs | **0** after time interpolation |
| Rows dropped | **0** — timeline preserved |
| Anomalies imputed | ~**248** intervals (Isolation Forest at `contamination=0.05`) |

Research cleaning profiles (`legacy_threshold`, `enhanced`) remain opt-in and are **not** the Phase 3 baseline until leadership reviews artifact diffs. See [Clean Dataset — Research profiles](clean-data.md#research-profiles) and [Anomaly Tuning Results](anomaly-tuning-results.md).

---

## Step 0: Codebase & State Review Gate

**Implemented** via `scripts/verify_phase2_state.py` (loads the clean CSV only — does not retrain Isolation Forest). See [Forecasting Baseline](forecasting-baseline.md).

| Check | Pass criterion |
|-------|----------------|
| **Pipeline integrity** | Clean artifact present (regenerate with `generate_clean_data.py` if needed) |
| **Data continuity** | Clean dataset has exactly **5,000** rows |
| **No NaNs** | Interpolation left **0** nulls in `Electricity_Consumed` |
| **Modularity** | Phase 3 can load cleaned data **without** re-triggering Phase 2 anomaly training loops by default |

**Warm-up note:** Rolling / lag feature warm-up may drop the first incomplete rows *when building model feature matrices*. That is separate from the clean artifact, which must stay **5,000** continuous rows with filled consumption.

---

## Forecasting Objective

Predict future values of `Electricity_Consumed` from:

- **Historical consumption** (lags, recent windows)
- **Exogenous context** where useful — weather columns and time-of-day / calendar features from Phase 1–2

Success means a documented, reproducible forecast on a held-out **future** window, with errors reported in metrics management can interpret.

---

## 1. Data Splitting (The Golden Rule)

Unlike tabular ML, we **cannot** use random shuffling (e.g. a naïve `train_test_split`). That causes **data leakage**: the model sees the future while learning to predict the past.

| Rule | Choice |
|------|--------|
| Method | Strict **chronological** split |
| Train | **70%** — learn patterns |
| Validation | **15%** — hyperparameters / early stopping |
| Test | **15%** — final unseen holdout for reported business value |

All model comparisons must use the same cut points so results stay fair.

---

## 2. Evaluation Metrics

Forecasting uses different scores than Phase 2 anomaly F1:

| Metric | Plain meaning | Role |
|--------|---------------|------|
| **MAE** (Mean Absolute Error) | Average absolute miss | Easy to explain to management |
| **RMSE** (Root Mean Squared Error) | Penalizes large misses more | Stress-tests bad spikes |
| **MAPE** (Mean Absolute Percentage Error) | Relative error | Useful when scale matters; **caution** if values approach zero |

Primary reporting stack: MAE + RMSE; MAPE as a secondary relative view with the zero-denominator caveat noted.

---

## 3. Model Progression

Build complexity sequentially. If a complex model cannot beat a simpler one on the same test window, we do not adopt it as the headline result.

### A. Naive baseline — **implemented**

| | |
|--|--|
| **What** | Predict that consumption equals the value from **24 hours earlier** at the same clock time — a **48-step** lag at 30-minute resolution |
| **Why** | If advanced models cannot beat this rule of thumb, they are not earning their complexity |
| **Code** | `naive_seasonal_forecast` + `scripts/evaluate_naive_baseline.py` — [Forecasting Baseline](forecasting-baseline.md) |

### B. Statistical baseline (Prophet) — **implemented**

| | |
|--|--|
| **What** | Univariate time-series model mapping trend and seasonality |
| **Why** | Strong mathematical floor; Prophet handles daily/weekly seasonality with less custom feature work |
| **Code** | `train_prophet_model` + `scripts/evaluate_prophet.py` — [Prophet Baseline](prophet-baseline.md) |

Auto-ARIMA remains deferred.

### C. Advanced ML (XGBoost) — **implemented**

| | |
|--|--|
| **What** | Gradient-boosted trees on tabular features |
| **How** | **Lag features** (\(t-1\), \(t-2\), \(t-48\)) plus temporal and weather columns from Phase 2 — see [XGBoost Prep](xgboost-prep.md) · [XGBoost Forecasting](xgboost-forecasting.md) |

### D. Deep learning (LSTM) — **implemented**

| | |
|--|--|
| **What** | Long Short-Term Memory network (recurrent architecture) |
| **How** | Sliding windows into 3D tensors `[samples, time_steps, features]` — past **12 hours** (24 steps) to predict the next **30 minutes** |
| **Code** | `create_sequences`, `EnergyLSTM`, `train_lstm_model`, `predict_lstm` — [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md) |

### E. Unified model comparison — **implemented**

| | |
|--|--|
| **What** | Single script runs Naive, Prophet, XGBoost, and LSTM on native test pipelines |
| **Why** | Research reporting — copy-paste Markdown metrics table and presentation PNG for grant write-ups |
| **Code** | `scripts/compare_forecasts.py` — [Forecast Model Comparison](forecast-model-comparison.md) |

### F. E2E pipeline consolidation — **implemented**

| | |
|--|--|
| **What** | Root `main.py` single CLI for the full Phase 1–3 workflow |
| **Scope** | argparse + logging → ingest → `build_all_features` → Isolation Forest detect → interpolate → chronological split → `--model` forecast → metrics → prediction CSV |
| **Code** | `main.py` — [E2E Pipeline](e2e-pipeline.md) |

`main.py` trains **one** model per run. For the four-model MAE/RMSE table and PNG, use `compare_forecasts.py` — [Forecast Model Comparison](forecast-model-comparison.md).

```mermaid
flowchart LR
  mainCli[main.py_CLI] --> ingest[load_smart_meter_data]
  ingest --> feats[build_all_features]
  feats --> detect[detect_anomalies_IF]
  detect --> cleanMem[interpolate_anomalies]
  cleanMem --> splitE2E[time_series_split]
  splitE2E --> route[run_selected_forecast]
  route --> evalExport[evaluate_and_export_CSV]
  clean[CleanCSV_5000] --> split[Chronological_70_15_15]
  split --> naive[Naive_48lag]
  split --> stats[Prophet]
  split --> xgb[XGBoost_lags]
  split --> lstm[LSTM_windows]
  naive --> compare[compare_forecasts.py]
  stats --> compare
  xgb --> compare
  lstm --> compare
  compare --> research[forecasting_research]
```

---

## 4. Educational & Research Deliverables

Once models are evaluated, technical iteration pauses and grant-facing documentation takes priority:

| Deliverable | Purpose |
|-------------|---------|
| [`docs/forecasting-research.md`](forecasting-research.md) | Research write-up: ladder winner (Prophet), weather vs history importance |
| [`notebooks/04_forecasting_tutorial.ipynb`](../notebooks/04_forecasting_tutorial.ipynb) · [Forecasting Tutorial](forecasting-tutorial.md) | Student-facing tutorial: chronological split, lags, XGBoost, metrics, plot |
| README + `requirements.txt` polish | Final dependency list and Phase 3 quick-start |
| Handover slide deck | Summary for the close-out meeting |

---

## Open Implementation Notes

### Shipped

- Naive seasonal floor, chronological split, and forecast metrics — [Forecasting Baseline](forecasting-baseline.md)
- Prophet statistical baseline — [Prophet Baseline](prophet-baseline.md)
- XGBoost lag prep and trainer — [XGBoost Prep](xgboost-prep.md) · [XGBoost Forecasting](xgboost-forecasting.md)
- LSTM sequence prep and trainer — [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md)
- Unified four-model comparison — [Forecast Model Comparison](forecast-model-comparison.md)
- E2E root CLI (`main.py`) — [E2E Pipeline](e2e-pipeline.md)
- Tutorial notebook and research write-up — [Forecasting Tutorial](forecasting-tutorial.md) · [Forecasting Research](forecasting-research.md)

### Still deferred

- Auto-ARIMA trainer
- Hyperparameter tuning for XGBoost, Prophet, and LSTM
- Handover slides
- Whether weather stays in the exogenous set after ablation-style checks (Phase 2 already showed weak linear weather signal for *anomaly* detection; forecasting may differ)

---

??? info "Technical deep dive"

    **Input API:** Load `data/processed/clean_smart_meter_data.csv`, or regenerate via `scripts/generate_clean_data.py` / `generate_clean_dataset(..., profile="legacy")`.

    **Split:** `time_series_split` — first 70% train, next 15% validation, final 15% test. Verify with `python -m src.data.make_forecast_dataset`.

    **Naive lag:** 48 steps = 24 h × 2 samples/hour — `naive_seasonal_forecast` in `train_forecast_models.py`.

    **Supervised lags (XGBoost prep):** `create_supervised_lags` in `build_features.py` — lags 1, 2, 48; verify with `python scripts/verify_xgboost_prep.py`.

    **LSTM sequences:** `create_sequences` in `build_features.py` — default window 24; verify with `python scripts/verify_lstm_prep.py`; score with `python scripts/evaluate_lstm.py`.

    **Metrics:** `evaluate_forecast` in `evaluate_forecast.py` (MAE / RMSE / MAPE on test for headline numbers).

    **Dependencies:** pandas, scikit-learn, `prophet>=1.1.5`, `xgboost>=2.0.0`, `torch>=2.0.0` (see `requirements.txt`).

    **Unified comparison:** `python scripts/compare_forecasts.py` — all four models, Markdown table, PNG asset.

    **E2E CLI:** `python main.py --model naive` (or `prophet` / `xgboost` / `lstm`) — full path through metrics and CSV export; `--save_clean_data` / `--output_path` optional; `--epochs` for LSTM.

    **Modularity:** Production forecast scripts load `clean_smart_meter_data.csv` from `generate_clean_data.py`. E2E `main.py` builds a clean frame in memory, splits chronologically, and trains one selected forecaster via `src/` helpers.

---

## References

- [Forecasting Baseline](forecasting-baseline.md) — clean-state gate, split, metrics, naive floor
- [Prophet Baseline](prophet-baseline.md) — Prophet trainer
- [XGBoost Prep](xgboost-prep.md) — supervised lags
- [XGBoost Forecasting](xgboost-forecasting.md) — XGBoost trainer
- [LSTM Prep](lstm-prep.md) — sequence tensors
- [LSTM Forecasting](lstm-forecasting.md) — LSTM trainer
- [Forecast Model Comparison](forecast-model-comparison.md) — unified ladder
- [E2E Pipeline](e2e-pipeline.md) — root CLI (ingest → forecast → metrics → CSV)
- [Forecasting Tutorial](forecasting-tutorial.md) — CMU educational notebook
- [Forecasting Research](forecasting-research.md) — research write-up
- [Clean Dataset](clean-data.md) — Phase 2 imputation artifact for Phase 3
- [Anomaly Detection](anomaly-detection.md) — Isolation Forest production path used for cleaning
- [Feature Engineering](feature-engineering.md) — temporal and rolling features to reuse / extend for lags
- [Architecture](architecture.md) — repository layout and Phase roadmap
- [Phase 2 Strategy](phase2-strategy.md) — detection planning that led to the clean handoff
- [Glossary](glossary.md) — shared term definitions

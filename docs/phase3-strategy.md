# Phase 3 Strategy — Time-Series Forecasting

Planning notes for the final technical phase: forecasting. Phase 1 (ingestion / EDA) and Phase 2 (anomaly detection and cleaning) are complete. We now have a clean, continuous dataset where historical anomalies have been masked and interpolated.

!!! success "Executive summary"

    - **Goal:** Predict future electricity use from the cleaned half-hour timeline — so demand forecasting sits on trustworthy history, not gaps or deleted rows.
    - **Starting point:** Default Phase 2 clean file (`clean_smart_meter_data.csv`) — continuous, ~248 repaired intervals, production recipe unchanged.
    - **Golden rule:** Split data **in time order** (70% train / 15% validation / 15% test). Never shuffle — that would leak the future into the past.
    - **Model ladder:** Beat a simple “same time yesterday” baseline before trusting Prophet/ARIMA, then XGBoost, then LSTM.
    - **Day 1–2 shipped:** Clean-state gate, chronological split, metrics module, and naive floor — see [Forecasting Baseline](forecasting-baseline.md).
    - **Prophet shipped (Day 3):** Univariate statistical baseline — see [Prophet Baseline](prophet-baseline.md).
    - **XGBoost shipped (Week 7):** Supervised lags + gradient-boosted regressor — see [XGBoost Prep](xgboost-prep.md) · [XGBoost Forecasting](xgboost-forecasting.md).
    - **LSTM shipped (Week 7 Days 3–5):** PyTorch sequence prep + `EnergyLSTM` — see [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md).
    - **Unified comparison shipped (Week 8 Day 1):** All four models via `compare_forecasts.py` — see [Forecast Model Comparison](forecast-model-comparison.md).
    - **E2E CLI shipped (Week 8 Days 2–3):** Root `main.py` wires ingest → features → Isolation Forest → in-memory clean — see [E2E Pipeline](e2e-pipeline.md).
    - **Terms:** [Glossary](glossary.md) — imputation, temporal split; forecasting metrics (MAE / RMSE / MAPE).

**Status:** Week 6 Day 1–3, Week 7, and Week 8 Day 1–3 **complete**; E2E Day 4 (forecast), research write-up, and tutorial notebook planned next  
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

**Implemented** via `scripts/verify_phase2_state.py` (loads the clean CSV only — does not retrain Isolation Forest). See [Forecasting Baseline — Step 0](forecasting-baseline.md#step-0--verify-phase-2-clean-state).

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

### E. Unified model comparison (Week 8) — **implemented**

| | |
|--|--|
| **What** | Single script runs Naive, Prophet, XGBoost, and LSTM on native test pipelines |
| **Why** | Research reporting — copy-paste Markdown metrics table and presentation PNG for grant write-ups |
| **Code** | `scripts/compare_forecasts.py` — [Forecast Model Comparison](forecast-model-comparison.md) |

### F. E2E pipeline consolidation (Week 8) — **in progress**

| | |
|--|--|
| **What** | Root `main.py` single CLI for the full Phase 1–3 workflow |
| **Day 2 done** | argparse, INFO logging, `load_smart_meter_data` + `build_all_features` |
| **Day 3 done** | `detect_anomalies` (Isolation Forest), `interpolate_anomalies`, optional `--save_clean_data` |
| **Day 4 planned** | Chronological split and `--model` forecasting |
| **Code** | `main.py` — [E2E Pipeline](e2e-pipeline.md) |

```mermaid
flowchart LR
  mainCli[main.py_CLI] --> ingest[load_smart_meter_data]
  ingest --> feats[build_all_features]
  feats --> detect[detect_anomalies_IF]
  detect --> cleanMem[interpolate_anomalies]
  cleanMem --> forecastLater[forecast_Day4_planned]
  clean[CleanCSV_5000] --> split[Chronological_70_15_15]
  split --> naive[Naive_48lag]
  split --> stats[Prophet]
  split --> xgb[XGBoost_lags]
  split --> lstm[LSTM_windows]
  naive --> compare[compare_forecasts.py]
  stats --> compare
  xgb --> compare
  lstm --> compare
  compare --> docs[forecasting_research_and_tutorial]
```

---

## 4. Educational & Research Deliverables

Once models are evaluated, technical iteration pauses and grant-facing documentation takes priority:

| Deliverable | Purpose |
|-------------|---------|
| [`docs/forecasting-research.md`](forecasting-research.md) *(planned)* | Research write-up: how predictable the load profile is, which features mattered, model limits |
| [`notebooks/04_forecasting_tutorial.ipynb`](../notebooks/04_forecasting_tutorial.ipynb) *(planned)* | Student-facing tutorial: chronological split and model progression |
| README + `requirements.txt` polish | Final dependency list and Phase 3 quick-start |
| Handover slide deck | Summary for the close-out meeting |

---

## Open Implementation Notes

**Done (Week 6 Day 1–2):** Step 0 audit script, `time_series_split`, `evaluate_forecast`, naive seasonal baseline — see [Forecasting Baseline](forecasting-baseline.md).

**Done (Week 6 Day 3):** Prophet trainer and `evaluate_prophet.py` — see [Prophet Baseline](prophet-baseline.md).

**Done (Week 7 Day 1–5):** `create_supervised_lags`, `train_xgboost_model`, `verify_xgboost_prep.py`, `evaluate_xgboost.py` — see [XGBoost Prep](xgboost-prep.md) · [XGBoost Forecasting](xgboost-forecasting.md). `create_sequences`, `EnergyLSTM`, `train_lstm_model`, `predict_lstm`, `verify_lstm_prep.py`, `evaluate_lstm.py` — see [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md).

**Done (Week 7 Day 3):** `create_sequences`, `verify_lstm_prep.py`, PyTorch dependency — see [LSTM Prep](lstm-prep.md).

**Done (Week 7 Days 3–5):** `create_sequences`, `EnergyLSTM`, `train_lstm_model`, `predict_lstm`, `verify_lstm_prep.py` — see [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md).

**Done (Week 8 Day 1):** `compare_forecasts.py`, `docs/assets/forecast_comparison.png` — see [Forecast Model Comparison](forecast-model-comparison.md).

**Done (Week 8 Day 2):** Root `main.py` — CLI + logging + ingestion + `build_all_features` — see [E2E Pipeline](e2e-pipeline.md).

**Done (Week 8 Day 3):** `detect_anomalies` (IF), `interpolate_anomalies`, `--save_clean_data` → `clean_pipeline_output.csv` — see [E2E Pipeline](e2e-pipeline.md).

**Model ladder complete.** E2E consolidation in progress (Day 4 forecasting). Still deferred:

- Whether weather stays in the exogenous set after ablation-style checks (Phase 2 already showed weak linear weather signal for *anomaly* detection; forecasting may differ)
- Auto-ARIMA trainer
- Hyperparameter tuning for XGBoost, Prophet, and LSTM

---

??? info "Technical deep dive"

    **Input API:** Load `data/processed/clean_smart_meter_data.csv`, or regenerate via `scripts/generate_clean_data.py` / `generate_clean_dataset(..., profile="legacy")`.

    **Split:** `time_series_split` — first 70% train, next 15% validation, final 15% test. Verify with `python -m src.data.make_forecast_dataset`.

    **Naive lag:** 48 steps = 24 h × 2 samples/hour — `naive_seasonal_forecast` in `train_forecast_models.py`.

    **Supervised lags (XGBoost prep):** `create_supervised_lags` in `build_features.py` — lags 1, 2, 48; verify with `python scripts/verify_xgboost_prep.py`.

    **LSTM sequences:** `create_sequences` in `build_features.py` — default window 24; verify with `python scripts/verify_lstm_prep.py`; score with `python scripts/evaluate_lstm.py`.

    **Metrics:** `evaluate_forecast` in `evaluate_forecast.py` (MAE / RMSE / MAPE on test for headline numbers).

    **Dependencies:** pandas, scikit-learn, `prophet>=1.1.5`, `xgboost>=2.0.0`, `torch>=2.0.0` (see `requirements.txt`).

    **LSTM prep:** `create_sequences` in `build_features.py` — verify with `python scripts/verify_lstm_prep.py`.

    **Unified comparison:** `python scripts/compare_forecasts.py` — all four models, Markdown table, PNG asset.

    **E2E CLI:** `python main.py` — Days 2–3 run ingest → features → IF detect → interpolate; `python main.py --save_clean_data` for optional checkpoint; `--model` / `--epochs` reserved for Day 4.

    **Modularity:** Production forecast scripts load `clean_smart_meter_data.csv` from `generate_clean_data.py`. E2E `main.py` builds a clean frame in memory (and optionally `clean_pipeline_output.csv`) via the same `interpolate_anomalies` helper.

---

## References

- [Forecasting Baseline](forecasting-baseline.md) — Week 6 Day 1–2 implementation notes
- [Prophet Baseline](prophet-baseline.md) — Week 6 Day 3 Prophet trainer
- [XGBoost Prep](xgboost-prep.md) — Week 7 Day 1 supervised lags
- [XGBoost Forecasting](xgboost-forecasting.md) — Week 7 Day 2 XGBoost trainer
- [LSTM Prep](lstm-prep.md) — Week 7 Day 3 sequence tensors
- [LSTM Forecasting](lstm-forecasting.md) — Week 7 Days 4–5 LSTM trainer
- [Forecast Model Comparison](forecast-model-comparison.md) — Week 8 Day 1 unified ladder
- [E2E Pipeline](e2e-pipeline.md) — Week 8 Days 2–3 root CLI (ingest → detect → clean)
- [Clean Dataset](clean-data.md) — Phase 2 imputation artifact for Phase 3
- [Anomaly Detection](anomaly-detection.md) — Isolation Forest production path used for cleaning
- [Feature Engineering](feature-engineering.md) — temporal and rolling features to reuse / extend for lags
- [Architecture](architecture.md) — repository layout and Phase roadmap
- [Phase 2 Strategy](phase2-strategy.md) — detection planning that led to the clean handoff
- [Glossary](glossary.md) — shared term definitions

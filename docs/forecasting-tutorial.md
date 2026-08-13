# Forecasting Tutorial — Phase 3, Week 9

CMU-Africa Techskills educational notebook that walks students from the **cleaned** smart-meter timeline through chronological splits, lag features, XGBoost training, test metrics, and an Actual vs Predicted chart.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Audience:** Students and contributors learning Phase 3 forecasting without running the full research ladder first.
- **Path:** Load clean CSV → chronological 70/15/15 split → lag demo → XGBoost fit → MAE/RMSE → ~3-day Actual vs Predicted plot.
- **Notebook:** [`notebooks/04_forecasting_tutorial.ipynb`](../notebooks/04_forecasting_tutorial.ipynb)
- **Terms:** [Glossary](glossary.md) — supervised lag features, MAE, RMSE, seasonal naive, E2E pipeline / main.py.

</div>

**Status:** Week 9 Days 1–2 complete  
**Builds on:** [Clean Dataset](clean-data.md), [Forecasting Baseline](forecasting-baseline.md), [XGBoost Forecasting](xgboost-forecasting.md), [E2E Pipeline](e2e-pipeline.md)

---

## How to open

Prerequisite: production clean artifact at `data/processed/clean_smart_meter_data.csv`. If missing:

```bash
python scripts/generate_clean_data.py
```

Then:

```bash
jupyter notebook notebooks/04_forecasting_tutorial.ipynb
```

Or open the file in VS Code / Cursor with a Jupyter kernel that has the project `.venv` selected (`pandas`, `matplotlib`, `xgboost`).

---

## Notebook outline

| Section | What students learn |
|---------|---------------------|
| **1. Setup & Load Clean Data** | Forecast from the interpolated production CSV — same artifact as Phase 3 scripts |
| **2. Chronological Splitting** | Why random splits leak the future; 70 / 15 / 15 train / val / test plot |
| **3. Lag Features** | `lag_1` and `lag_48` via explicit `shift` on a small train window |
| **4. Training XGBoost** | Full `create_supervised_lags` matrix, re-split, `XGBRegressor` fit |
| **5. Scoring (MAE & RMSE)** | Plain-English metrics; score held-out test predictions |
| **6. What This Notebook Achieved** | Summary + pointers to `compare_forecasts.py` and `main.py` |

The notebook intentionally focuses on **XGBoost** as the tabular teaching model. For the four-model research ladder (naive / Prophet / XGBoost / LSTM), use [Forecast Model Comparison](forecast-model-comparison.md). For the consolidating CLI (ingest → detect → clean → forecast → export), use [E2E Pipeline](e2e-pipeline.md). For the grant-facing findings narrative (Prophet winner; weather vs history), continue to [Forecasting Research](forecasting-research.md).

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Aligned helpers:** `create_supervised_lags` from `src.features.build_features`; metric helpers from `src.models.evaluate_forecast`; feature column list matches `scripts/evaluate_xgboost.py`.

**Hyperparams:** `XGBRegressor(n_estimators=100, learning_rate=0.1)` with validation `eval_set` — same defaults as `train_xgboost_model`.

**Plot window:** first **144** test steps (~3 days at 30-minute resolution), matching `compare_forecasts.py`.

</details>

---

## References

- [E2E Pipeline](e2e-pipeline.md) — root `main.py` consolidating CLI
- [XGBoost Forecasting](xgboost-forecasting.md) — research trainer and scores
- [Forecasting Research](forecasting-research.md) — Phase 3 findings write-up
- [Forecasting Baseline](forecasting-baseline.md) — chronological split protocol
- [Forecast Model Comparison](forecast-model-comparison.md) — four-model ladder
- [Phase 3 Strategy](phase3-strategy.md) — educational deliverables
- [Getting Started](getting-started.md) — environment setup
- [Glossary](glossary.md) — MAE, RMSE, supervised lag features, feature importance (gain)

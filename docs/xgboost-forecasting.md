# XGBoost Forecasting — Phase 3, Week 7 (Day 2)

Working notes for training and scoring a gradient-boosted **XGBRegressor** on tabular lag and context features, compared against naive and Prophet floors.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **What we added:** XGBoost regressor on **9 tabular features** (3 consumption lags + weather + calendar columns).
- **Validation monitoring:** Training uses `eval_set=[(X_val, y_val)]` — validation loss is tracked; **test** metrics are reported for business comparison.
- **Beats naive floor:** Example test MAE ≈ **0.125**, RMSE ≈ **0.154** vs naive **0.171** / **0.214**; slightly above Prophet **0.121** / **0.149** on this run.
- **Same timeline:** Lag warm-up → chronological 70/15/15 split → `evaluate_forecast` on held-out test rows.
- **Terms:** [Glossary](glossary.md) — XGBoost, supervised lag features, eval_set, MAE, RMSE.

</div>

**Status:** Week 7 Day 2 complete — XGBoost trainer and evaluation script  
**Modules:** `src/models/train_forecast_models.py`, `src/features/build_features.py`, `src/models/evaluate_forecast.py`  
**Scripts:** `scripts/evaluate_xgboost.py`  
**Builds on:** [XGBoost Prep](xgboost-prep.md), [Prophet Baseline](prophet-baseline.md), [Forecasting Baseline](forecasting-baseline.md)

---

## End-to-End Pipeline

```text
clean CSV (5000 rows)
  → create_supervised_lags          # 4952 rows after warm-up
  → time_series_split (70/15/15)    # train / val / test in time order
  → train_xgboost_model(X_train, y_train, X_val, y_val)
  → model.predict(X_test)
  → evaluate_forecast(y_test, y_pred)
```

```mermaid
flowchart LR
  clean[CleanCSV] --> lags[create_supervised_lags]
  lags --> split[time_series_split]
  split --> train[train_xgboost_model]
  train --> predict[predict_test]
  predict --> metrics[evaluate_forecast]
  metrics --> compare[Compare_naive_Prophet_floors]
```

---

## Feature Matrix (9 columns)

Defined in `scripts/evaluate_xgboost.py` — must match trainer input exactly:

| Column | Source |
|--------|--------|
| `Electricity_Consumed_lag_1` | [XGBoost Prep](xgboost-prep.md) |
| `Electricity_Consumed_lag_2` | [XGBoost Prep](xgboost-prep.md) |
| `Electricity_Consumed_lag_48` | [XGBoost Prep](xgboost-prep.md) |
| `Temperature` | Clean artifact |
| `Humidity` | Clean artifact |
| `hour` | Phase 2 temporal features |
| `day_of_week` | Phase 2 temporal features |
| `month` | Phase 2 temporal features |
| `is_weekend` | Phase 2 temporal features |

Target column (not in `X`): `Electricity_Consumed` at time **t**.

---

## API — `train_xgboost_model`

Module: `src/models/train_forecast_models.py`

```python
train_xgboost_model(X_train, y_train, X_val, y_val) -> XGBRegressor
```

| Parameter | Value |
|-----------|-------|
| Estimator | `XGBRegressor(n_estimators=100, learning_rate=0.1)` |
| Monitoring | `eval_set=[(X_val, y_val)]`, `verbose=False` |
| Dependency | `xgboost>=2.0.0` |

Validation rows inform training loss only — **reported business metrics** come from the held-out **test** split via `evaluate_xgboost.py`.

---

## Score the Model

```bash
python scripts/evaluate_xgboost.py
```

On Windows:

```powershell
.venv\Scripts\activate
python scripts/evaluate_xgboost.py
```

The script compares test MAE/RMSE against documented **naive** and **Prophet** floor constants and prints PASS/NOTE lines.

### Example run (local, reproducible)

Chronological split sizes **after lag warm-up**:

| Split | Rows |
|-------|------|
| Train | 3,466 |
| Validation | 743 |
| Test | 743 |

Test window: `2024-03-29 16:30:00` → `2024-04-14 03:30:00`

| Metric | XGBoost | Naive floor | Prophet floor | LSTM floor |
|--------|---------|-------------|---------------|------------|
| MAE | **0.125274** | 0.171150 | 0.121071 | 0.122156 |
| RMSE | **0.153876** | 0.214034 | 0.148670 | 0.151200 |
| MAPE | Unstable on near-zero true values — use MAE/RMSE | — | — | — |

**Interpretation:** XGBoost beats the naive floor on MAE and RMSE but is slightly above Prophet on this default hyperparameter run — reasonable for a first-pass trainer without tuning.

Re-run after regenerating the clean artifact or changing features; numbers may shift.

---

## What's Next

Per [Phase 3 Strategy](phase3-strategy.md):

1. ~~LSTM~~ sliding windows → [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md)
2. ~~Unified model comparison~~ → [Forecast Model Comparison](forecast-model-comparison.md)
3. Hyperparameter tuning for XGBoost (`n_estimators`, `learning_rate`, feature ablation)
4. Research write-up and tutorial notebook (`forecasting-research.md`, `04_forecasting_tutorial.ipynb`)
5. Auto-ARIMA (deferred)

Each model must use the same chronological cut and `evaluate_forecast` helpers.

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Why validation but report test:** `eval_set` lets XGBoost track validation loss during `fit` without peeking at the final test window. Reported scores always come from `y_test` vs `y_pred`.

**Lag warm-up vs split:** Raw clean CSV has 5,000 rows; after dropping 48 lag-incomplete rows, `time_series_split` sees 4,952 rows — so train/val/test counts differ from the naive/Prophet scripts that split before lags.

**Baseline floor constants** (hardcoded reference in `evaluate_xgboost.py`):

- Naive: MAE **0.171150**, RMSE **0.214034**
- Prophet: MAE **0.121071**, RMSE **0.148670**
- LSTM: MAE **0.122156**, RMSE **0.151200**

**Unified comparison:** `python scripts/compare_forecasts.py` — see [Forecast Model Comparison](forecast-model-comparison.md).

**Commands:**

```bash
python scripts/verify_xgboost_prep.py
python scripts/evaluate_xgboost.py
```

**Modularity:** Loads clean CSV only; does not retrain anomaly models unless you run `generate_clean_data.py` explicitly.

</details>

---

## References

- [XGBoost Prep](xgboost-prep.md) — supervised lag feature engineering
- [LSTM Forecasting](lstm-forecasting.md) — recurrent baseline
- [Forecast Model Comparison](forecast-model-comparison.md) — unified ladder scoring
- [Prophet Baseline](prophet-baseline.md) — statistical floor
- [Forecasting Baseline](forecasting-baseline.md) — gate, split, metrics
- [Feature Engineering](feature-engineering.md) — Phase 2 temporal columns reused here
- [Phase 3 Strategy](phase3-strategy.md) — model ladder
- [LSTM Forecasting](lstm-forecasting.md) — sequence-based deep learning baseline
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — XGBoost, eval_set, supervised lags
- [Getting Started](getting-started.md) — install and Phase 3 commands

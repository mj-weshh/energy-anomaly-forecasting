# Prophet Baseline — Phase 3, Week 6 (Day 3)

Working notes for the first **statistical** forecast baseline: Facebook Prophet on univariate consumption, scored on the same chronological test window as the naive seasonal floor.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **What we added:** Prophet fits trend and seasonality from timestamps alone — no manual lag columns required.
- **Same rules:** Uses the Phase 2 clean CSV and **70/15/15** chronological split from [Forecasting Baseline](forecasting-baseline.md).
- **Beats the naive floor:** Example test MAE ≈ **0.121**, RMSE ≈ **0.149** vs naive **0.171** / **0.214**.
- **Dependency:** `prophet>=1.1.5` in `requirements.txt` — install via [Getting Started](getting-started.md).
- **Terms:** [Glossary](glossary.md) — Prophet, MAE, RMSE, forecast chronological split.

</div>

**Status:** Week 6 Day 3 complete — Prophet trainer and evaluation script  
**Modules:** `src/models/train_forecast_models.py`, `src/models/evaluate_forecast.py`, `src/data/make_forecast_dataset.py`  
**Scripts:** `scripts/evaluate_prophet.py`  
**Builds on:** [Forecasting Baseline](forecasting-baseline.md), [Phase 3 Strategy](phase3-strategy.md)

---

## Why Prophet Here

Statistical models like Prophet map **datetime** and **seasonality** natively. Tree models like XGBoost treat each row as a flat feature vector — they need explicit **lag columns** (see [XGBoost Prep](xgboost-prep.md)).

| Approach | Time representation |
|----------|---------------------|
| Prophet / ARIMA | Native `ds` datetime + built-in seasonal components |
| XGBoost | Lag columns + optional temporal features from Phase 2 |

Prophet establishes a strong **mathematical floor** between the naive rule-of-thumb and gradient-boosted tabular models.

---

## API — `train_prophet_model`

Module: `src/models/train_forecast_models.py`

```python
train_prophet_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray
```

| Step | Behaviour |
|------|-----------|
| Column mapping | `Timestamp` → `ds`, `Electricity_Consumed` → `y` (internal copies only) |
| Fit | Default `Prophet()` on **train** rows only |
| Predict | `yhat` for every row in **test** (chronological horizon) |
| Validation split | **Not used** by this trainer — unlike XGBoost `eval_set` |

Returns a 1-D `numpy` array of forecasts aligned to `test_df` row order.

---

## Score the Baseline

```bash
python scripts/evaluate_prophet.py
```

Workflow: load clean CSV → `time_series_split` → `train_prophet_model` → `evaluate_forecast` on test → compare to documented naive floor constants.

On Windows, use the project virtual environment:

```powershell
.venv\Scripts\activate
python scripts/evaluate_prophet.py
```

The script includes a venv guard when Prophet is missing from the active interpreter. A harmless **Plotly import warning** may appear — CLI scoring still works.

### Example run (local, reproducible)

| Metric | Value |
|--------|-------|
| MAE | **0.121071** |
| RMSE | **0.148670** |
| MAPE | Unstable on near-zero true values — use MAE/RMSE for comparisons |

Split sizes (full 5,000-row clean CSV, before lag warm-up):

| Split | Rows |
|-------|------|
| Train | 3,500 |
| Validation | 750 (not used for Prophet scoring) |
| Test | 750 |

Re-run after regenerating the clean artifact; numbers may shift slightly if cleaning or data path changes.

---

## What's Next

1. **XGBoost prep** — supervised lag features for tabular forecasting → [XGBoost Prep](xgboost-prep.md)
2. **XGBoost training** — gradient-boosted trees on lags + temporal/weather columns → [XGBoost Forecasting](xgboost-forecasting.md)
3. ~~**LSTM** sliding windows~~ — **done:** [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md)
4. Auto-ARIMA remains deferred

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Script:** `evaluate_prophet.py` — end-to-end Prophet score with naive floor comparison.

**Naive floor constants (reference):** MAE **0.171150**, RMSE **0.214034** — same as [Forecasting Baseline](forecasting-baseline.md).

**Modularity:** Loads the clean CSV only; does not retrain Isolation Forest unless you run `generate_clean_data.py` explicitly.

**Commands:**

```bash
python scripts/generate_clean_data.py   # if artifact missing
python scripts/verify_phase2_state.py
python scripts/evaluate_prophet.py
```

</details>

---

## References

- [Forecasting Baseline](forecasting-baseline.md) — gate, split, metrics, naive floor
- [XGBoost Prep](xgboost-prep.md) — supervised lags for tree models
- [Phase 3 Strategy](phase3-strategy.md) — model ladder
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — Prophet, MAE, RMSE
- [Getting Started](getting-started.md) — install and Phase 3 commands

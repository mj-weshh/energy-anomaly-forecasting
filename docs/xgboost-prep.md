# XGBoost Prep — Phase 3, Week 7 (Day 1)

Working notes for converting the continuous clean timeline into a **supervised tabular** frame before XGBoost training.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Why lags:** XGBoost does not read clock time — each row must carry past consumption as explicit columns beside the target at time **t**.
- **Lags added:** `t-1` (30 min), `t-2` (1 h), `t-48` (24 h at 30-minute resolution).
- **Warm-up drop:** First **48** rows removed so every row has complete lag history → **4,952** rows on the default clean CSV.
- **Verify:** `python scripts/verify_xgboost_prep.py` prints shape, columns, and a sample to confirm alignment.
- **Terms:** [Glossary](glossary.md) — supervised lag features, forecast chronological split.

</div>

**Status:** Week 7 Day 1 complete — supervised lag function and verification script  
**Module:** `src/features/build_features.py` — `create_supervised_lags`  
**Script:** `scripts/verify_xgboost_prep.py`  
**Builds on:** [Forecasting Baseline](forecasting-baseline.md), [Prophet Baseline](prophet-baseline.md), [Feature Engineering](feature-engineering.md)

---

## Tree Models vs Time Series

Statistical models like Prophet map timestamps natively. Tree models like **XGBoost** treat each row as a flat feature vector — they need **past values on the same row** as predictors for the target at time **t**.

| Model family | Time representation |
|--------------|---------------------|
| Prophet / ARIMA | Native datetime / seasonality |
| XGBoost | Lag columns + optional temporal features from Phase 2 |

---

## API — `create_supervised_lags`

Module: `src/features/build_features.py`

```python
create_supervised_lags(df, target_col="Electricity_Consumed") -> pd.DataFrame
```

| Output column | Meaning |
|---------------|---------|
| `{target_col}_lag_1` | Value at **t − 1** step (30 minutes earlier) |
| `{target_col}_lag_2` | Value at **t − 2** steps (1 hour earlier) |
| `{target_col}_lag_48` | Value at **t − 48** steps (24 hours earlier) |

Rows are sorted by `Timestamp` when present. After shifting, rows with missing lag history are **dropped** and the index is reset.

XGBoost can tolerate NaNs, but we drop incomplete rows so **all forecast models evaluate on the same timeline** after feature warm-up. On the default clean artifact (`5000` continuous rows), output shape is **4952 × 18** (original clean columns plus three lags).

---

## Verify Tabular Prep

```bash
python scripts/verify_xgboost_prep.py
```

Expect printed confirmation of:

- Input shape `(5000, …)` from the clean CSV
- Output shape `(4952, 18)` after lag warm-up
- Presence of `Electricity_Consumed_lag_1`, `_lag_2`, `_lag_48`
- Sample rows showing target and lag alignment

---

## Relationship to Phase 2 Features

The clean artifact already includes temporal and rolling columns from production cleaning (`hour`, `day_of_week`, `month`, `is_weekend`, etc.). Lag prep adds consumption history columns; those temporal features are reused as XGBoost inputs in [XGBoost Forecasting](xgboost-forecasting.md).

**Module map:** `create_supervised_lags` lives alongside Phase 2 helpers in `build_features.py`. Day 1 verify uses the clean CSV directly; full XGBoost training chains `create_supervised_lags` → `time_series_split` → `train_xgboost_model`.

---

## What's Next

1. ~~XGBoost training and evaluation~~ — see [XGBoost Forecasting](xgboost-forecasting.md) (Week 7 Day 2 complete)
2. ~~Compare MAE / RMSE against naive and Prophet floors~~ — see [Forecast Model Comparison](forecast-model-comparison.md)
3. ~~LSTM~~ sliding windows → [LSTM Prep](lstm-prep.md) · [LSTM Forecasting](lstm-forecasting.md)
4. Hyperparameter tuning and feature ablation (deferred)

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Verify command:**

```bash
python scripts/verify_xgboost_prep.py
```

**Split note:** After lags, `time_series_split` operates on **4,952** rows — not the raw 5,000. Train/val/test counts shrink proportionally (see [XGBoost Forecasting](xgboost-forecasting.md) for example sizes).

**Chronological order:** Never shuffle after lag creation — future values must not appear in lag columns for past rows.

</details>

---

## References

- [XGBoost Forecasting](xgboost-forecasting.md) — trainer, evaluation script, example metrics
- [Prophet Baseline](prophet-baseline.md) — statistical floor Prophet sets
- [Forecasting Baseline](forecasting-baseline.md) — gate and split
- [Feature Engineering](feature-engineering.md) — Phase 2 temporal and rolling features
- [Phase 3 Strategy](phase3-strategy.md) — model ladder and XGBoost plan
- [Glossary](glossary.md) — supervised lag features

# LSTM Prep — Phase 3, Week 7 (Day 3)

Working notes for converting the clean timeline into **3D sequence tensors** before LSTM training with PyTorch.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Framework:** **PyTorch** (`torch>=2.0.0`) chosen for deep-learning forecasting — architectural control over the LSTM stack.
- **Why 3D:** LSTMs read **sequences**, not single flat rows — input shape `[samples, time_steps, features]`.
- **Default window:** **24 steps** = **12 hours** of history at 30-minute resolution per sample.
- **Feature matrix:** 7 columns (consumption + weather + calendar fields from the clean artifact).
- **Verify:** `python scripts/verify_lstm_prep.py` — example tensor shape `(176, 24, 7)` on a 200-row slice.
- **Terms:** [Glossary](glossary.md) — LSTM, PyTorch, sliding window / sequence tensor.

</div>

**Status:** Week 7 Day 3 complete — sliding-window sequences and verification script  
**Module:** `src/features/build_features.py` — `create_sequences`  
**Script:** `scripts/verify_lstm_prep.py`  
**Builds on:** [Forecasting Baseline](forecasting-baseline.md), [Feature Engineering](feature-engineering.md), [XGBoost Forecasting](xgboost-forecasting.md)

---

## Recurrent Models vs Tabular Trees

XGBoost uses **explicit lag columns** on each row ([XGBoost Prep](xgboost-prep.md)). LSTMs consume a **contiguous window** of past timesteps as a single 3D tensor — the network learns temporal patterns internally rather than relying on hand-picked lag indices alone.

| Model family | Time representation |
|--------------|---------------------|
| Prophet / ARIMA | Native datetime / seasonality |
| XGBoost | Lag columns + temporal features |
| LSTM | Sliding window tensor `[samples, seq_len, features]` |

---

## Feature Matrix (7 columns)

Defined in `scripts/verify_lstm_prep.py` and reused in `scripts/compare_forecasts.py` — column order must match the trainer:

| Column | Source |
|--------|--------|
| `Electricity_Consumed` | Clean artifact (target column at index 0) |
| `Temperature` | Clean artifact |
| `Humidity` | Clean artifact |
| `hour` | Phase 2 temporal features |
| `day_of_week` | Phase 2 temporal features |
| `month` | Phase 2 temporal features |
| `is_weekend` | Phase 2 temporal features |

---

## API — `create_sequences`

Module: `src/features/build_features.py`

```python
create_sequences(data, seq_length=24) -> tuple[np.ndarray, np.ndarray]
```

| Input | Shape / meaning |
|-------|-----------------|
| `data` | 2D array `(n_timesteps, n_features)` in chronological order |
| `seq_length` | Past window length (default **24** steps) |

| Output | Shape / meaning |
|--------|-----------------|
| `X` | `(num_samples, seq_length, n_features)` — input windows |
| `y` | `(num_samples, n_features)` — **next** row after each window |

With `num_samples = n_timesteps - seq_length`. On the default clean artifact (**5,000** rows), full-series output is **4,976** samples.

Each sample uses the **observed** rows in its window; targets are the row immediately following the window (one-step-ahead forecast).

---

## Verify Sequence Prep

```bash
python scripts/verify_lstm_prep.py
```

On Windows:

```powershell
.venv\Scripts\activate
python scripts/verify_lstm_prep.py
```

Expect printed confirmation of:

- Input 2D shape `(200, 7)` (first 200 rows of clean CSV)
- NumPy `X` shape `(176, 24, 7)` and matching `torch.float32` tensor
- **PASS** — 3D sequence tensor ready for LSTM input

Requires `torch>=2.0.0` in the project `.venv`.

---

## Split Semantics (Important)

Unlike XGBoost (which calls `time_series_split` on a tabular frame after lag warm-up), the LSTM evaluation and comparison paths:

1. Build sequences on the **full** chronological feature matrix first.
2. Split `(X, y)` arrays by index using the same **70 / 15 / 15** fraction math as `time_series_split`.

Do **not** split the raw DataFrame before `create_sequences` — that would break window continuity at split boundaries.

Full training and scoring: [LSTM Forecasting](lstm-forecasting.md) · unified comparison: [Forecast Model Comparison](forecast-model-comparison.md).

---

## Relationship to Phase 2 Features

The clean artifact already includes temporal columns from production cleaning. LSTM prep stacks them with consumption and weather into a multivariate window — no additional lag columns are required at this stage.

**Module map:** `create_sequences` lives alongside Phase 2 helpers and `create_supervised_lags` in `build_features.py`.

---

## What's Next

1. ~~**LSTM architecture, training, and inference**~~ — **done:** [LSTM Forecasting](lstm-forecasting.md)
2. ~~**Unified model comparison**~~ — **done:** [Forecast Model Comparison](forecast-model-comparison.md)
3. ~~Tutorial notebook~~ — **done:** [Forecasting Tutorial](forecasting-tutorial.md)
4. ~~Research write-up~~ — **done:** [Forecasting Research](forecasting-research.md)

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Verify command:**

```bash
python scripts/verify_lstm_prep.py
```

**Full-series sample count:** `5000 - 24 = 4976` sequence samples when using all clean rows.

**Tensor dtype:** Verification converts `X` to `torch.float32`; `make_lstm_dataloader` uses the same dtype for training.

**Chronological order:** Never shuffle sequence arrays before splitting — future values must not appear inside past windows.

**Dependency:** `torch>=2.0.0` in `requirements.txt`.

</details>

---

## References

- [LSTM Forecasting](lstm-forecasting.md) — `EnergyLSTM`, training loop, `predict_lstm`
- [Forecast Model Comparison](forecast-model-comparison.md) — side-by-side scoring of all four models
- [XGBoost Prep](xgboost-prep.md) — tabular lag alternative for tree models
- [Forecasting Baseline](forecasting-baseline.md) — gate and split fractions
- [Feature Engineering](feature-engineering.md) — Phase 2 temporal columns reused here
- [Phase 3 Strategy](phase3-strategy.md) — model ladder
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — LSTM, PyTorch, sliding window
- [Getting Started](getting-started.md) — install and Phase 3 commands

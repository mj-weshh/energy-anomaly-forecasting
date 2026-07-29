# LSTM Prep — Phase 3, Week 7 (Day 3)

Working notes for converting the clean timeline into **3D sequence tensors** before LSTM training with PyTorch.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **Framework:** **PyTorch** (`torch>=2.0.0`) chosen for deep-learning forecasting — architectural control over the LSTM stack.
- **Why 3D:** LSTMs read **sequences**, not single flat rows — input shape `[samples, time_steps, features]`.
- **Default window:** **24 steps = 12 hours** at 30-minute resolution; each sample predicts the **next** row.
- **Verify:** `python scripts/verify_lstm_prep.py` confirms tensor shape `(176, 24, 7)` on a 200-row slice.
- **Terms:** [Glossary](glossary.md) — sliding window, PyTorch, LSTM prep, sequence tensor.

</div>

**Status:** Week 7 Day 3 complete — sliding-window generator and tensor verification script  
**Module:** `src/features/build_features.py` — `create_sequences`  
**Script:** `scripts/verify_lstm_prep.py`  
**Builds on:** [XGBoost Forecasting](xgboost-forecasting.md), [Feature Engineering](feature-engineering.md), [Phase 3 Strategy](phase3-strategy.md)

---

## Tabular vs Sequence Models

| Model family | Data shape | How history is represented |
|--------------|------------|----------------------------|
| XGBoost | 2D tabular | Lag columns on one row — see [XGBoost Prep](xgboost-prep.md) |
| LSTM | 3D sequences | Sliding window of past rows per sample |

XGBoost asks: “What were consumption and context **at fixed lags** on this row?”  
LSTM asks: “What happened over the **last N consecutive intervals** before this prediction?”

---

## API — `create_sequences`

Module: `src/features/build_features.py`

```python
create_sequences(data: np.ndarray, seq_length: int = 24) -> tuple[np.ndarray, np.ndarray]
```

| Argument | Meaning |
|----------|---------|
| `data` | 2D array `(n_timesteps, n_features)` in chronological order |
| `seq_length` | Window length (default **24** = 12 hours) |

For each index `i` from `0` to `n_timesteps - seq_length - 1`:

- `X[i] = data[i : i + seq_length]` → shape `(seq_length, n_features)`
- `y[i] = data[i + seq_length]` → shape `(n_features,)`

Returns:

| Array | Shape |
|-------|-------|
| `X` | `(num_samples, seq_length, n_features)` |
| `y` | `(num_samples, n_features)` |

with `num_samples = n_timesteps - seq_length`.

**Validation:** raises `ValueError` if `data` is not 2D, `seq_length < 1`, or the series is too short.

---

## Feature Columns (verify script)

From `scripts/verify_lstm_prep.py` — seven numeric columns on the clean artifact:

`Electricity_Consumed`, `Temperature`, `Humidity`, `hour`, `day_of_week`, `month`, `is_weekend`

Full-series LSTM training will reuse or extend this set when the trainer lands.

---

## Verify Sequences and Tensors

```bash
python scripts/verify_lstm_prep.py
```

On Windows:

```powershell
.venv\Scripts\activate
python scripts/verify_lstm_prep.py
```

Workflow: load clean CSV → take first **200** chronological rows → build 2D feature matrix → `create_sequences` → `torch.tensor(X, dtype=torch.float32)` → assert shape → **PASS**.

The script includes a venv guard when PyTorch is missing from the active interpreter.

### Example run (local, reproducible)

| Artifact | Shape |
|----------|-------|
| Input 2D | `(200, 7)` |
| NumPy `X` | `(176, 24, 7)` |
| NumPy `y` | `(176, 7)` |
| PyTorch `X_tensor` | `(176, 24, 7)` |

Sample count: `176 = 200 - 24`.

Exit line: `PASS — 3D sequence tensor ready for LSTM input.`

---

## What's Next

1. **LSTM model architecture** and training loop (deferred)
2. Chronological train/val/test wiring for sequence batches
3. Score against naive / Prophet / XGBoost floors with `evaluate_forecast`
4. Research write-up and tutorial notebook (`forecasting-research.md`, `04_forecasting_tutorial.ipynb`)

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Dependency:** `torch>=2.0.0` in `requirements.txt` — install via [Getting Started](getting-started.md).

**No split yet:** Day 3 stops at sequence + tensor verification. Chronological splitting of sequence samples will ship with the LSTM trainer.

**Tensor dtype:** Verification uses `torch.float32` to match typical GPU/CPU training defaults.

**Commands:**

```bash
pip install -r requirements.txt
python scripts/verify_lstm_prep.py
```

</details>

---

## References

- [XGBoost Prep](xgboost-prep.md) — tabular lag alternative
- [XGBoost Forecasting](xgboost-forecasting.md) — prior model in the ladder
- [Feature Engineering](feature-engineering.md) — Phase 2 columns reused as LSTM features
- [Phase 3 Strategy](phase3-strategy.md) — model ladder and PyTorch decision
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — sliding window, PyTorch, LSTM prep
- [Getting Started](getting-started.md) — install and Phase 3 commands

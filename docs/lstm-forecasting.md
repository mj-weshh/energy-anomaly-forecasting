# LSTM Forecasting — Phase 3, Week 7 (Days 4–5)

Working notes for the PyTorch LSTM regressor: architecture, training loop, test-set inference, and scoring against the Phase 3 model ladder.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **What we added:** `EnergyLSTM` — compact LSTM on **7 multivariate features**, 24-step windows, scalar consumption forecast.
- **Training:** Adam + MSE for **20 epochs**; validation loss printed each epoch to monitor overfitting.
- **Inference:** `predict_lstm` detaches PyTorch outputs to NumPy for sklearn-compatible metrics.
- **Beats naive and XGBoost floors:** Example test MAE ≈ **0.122**, RMSE ≈ **0.151** vs naive **0.171** / **0.214** and XGBoost **0.125** / **0.154**; slightly above Prophet **0.121** / **0.149** on this default run.
- **Metrics scale:** MAE/RMSE are on **normalized** consumption (0–1); no inverse transform in this pipeline.
- **Terms:** [Glossary](glossary.md) — EnergyLSTM, PyTorch DataLoader, normalized forecast metrics.

</div>

**Status:** Week 7 Days 4–5 complete — architecture, training, inference  
**Modules:** `src/models/lstm_model.py`, `src/models/train_forecast_models.py`, `src/models/evaluate_forecast.py`  
**Scripts:** `scripts/compare_forecasts.py` (unified comparison; trains LSTM as part of the full ladder)  
**Builds on:** [LSTM Prep](lstm-prep.md), [Forecasting Baseline](forecasting-baseline.md), [XGBoost Forecasting](xgboost-forecasting.md)

---

## End-to-End Pipeline

```text
clean CSV (5000 rows)
  → 7-column feature matrix
  → create_sequences                    # 4976 samples after 24-step warm-up
  → chronological split on (X, y)       # 70/15/15 on sequence samples
  → make_lstm_dataloader                # batch_size=32, shuffle=False
  → EnergyLSTM
  → train_lstm_model (20 epochs)
  → predict_lstm
  → evaluate_forecast(y_true, y_pred)
```

```mermaid
flowchart LR
  clean[CleanCSV] --> seq[create_sequences]
  seq --> split[sequence_split_70_15_15]
  split --> loader[make_lstm_dataloader]
  loader --> train[train_lstm_model]
  train --> predict[predict_lstm]
  predict --> metrics[evaluate_forecast]
  metrics --> compare[Forecast_Model_Comparison]
```

---

## Architecture — `EnergyLSTM`

Module: `src/models/lstm_model.py`

| Layer | Configuration |
|-------|---------------|
| `nn.LSTM` | `input_size=7`, `hidden_size=64`, `num_layers=1`, `batch_first=True` |
| `nn.Linear` | `64 → 1` on the **final** LSTM timestep |
| Output | Shape `(batch,)` — one `Electricity_Consumed` value per sequence |

Forward pass: `lstm_out[:, -1, :]` → fully connected head → squeeze to 1-D predictions.

---

## Training and Inference APIs

Module: `src/models/train_forecast_models.py`

### `make_lstm_dataloader`

```python
make_lstm_dataloader(X, y, batch_size=32, consumption_index=0, shuffle=False)
```

Wraps `TensorDataset` + `DataLoader`. When `y` has shape `(N, n_features)`, only column **`consumption_index`** (default **0** = `Electricity_Consumed`) is used as the scalar target.

### `train_lstm_model`

```python
train_lstm_model(model, train_loader, val_loader, epochs=20, learning_rate=1e-3)
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | MSE (`nn.MSELoss`) |
| Device | CUDA when available, else CPU |
| Monitoring | Mean train + val loss printed per epoch |

Returns the trained model in place.

### `predict_lstm`

```python
predict_lstm(model, test_loader) -> np.ndarray
```

- Sets `model.eval()` and runs under `torch.no_grad()`.
- Moves batches to the model's device; collects `outputs.detach().cpu().numpy()`.
- Returns a flat 1-D array aligned with test loader order (`shuffle=False`).

Actuals for scoring come from `y_test[:, 0]`, not from loader targets during inference.

---

## Scoring

LSTM test metrics are produced by the unified comparison script:

```bash
python scripts/compare_forecasts.py
```

On Windows:

```powershell
.venv\Scripts\activate
python scripts/compare_forecasts.py
```

Training runs **20 epochs** (~1–2 minutes on CPU) as part of the four-model pipeline.

### Example run (local, reproducible)

Chronological split sizes **after sequence warm-up**:

| Split | Sequences |
|-------|-----------|
| Train | 3,483 |
| Validation | 746 |
| Test | 747 |

| Metric | LSTM | Naive floor | Prophet floor | XGBoost floor |
|--------|------|-------------|---------------|---------------|
| MAE | **0.122156** | 0.171150 | 0.121071 | 0.125274 |
| RMSE | **0.151200** | 0.214034 | 0.148670 | 0.153876 |

**Interpretation:** LSTM beats the naive and XGBoost floors on MAE and RMSE but is slightly above Prophet on this first-pass hyperparameter run — reasonable without tuning or early stopping.

Re-run after regenerating the clean artifact or changing features; numbers may shift.

---

## Normalized Metrics

The Kaggle clean artifact stores consumption and weather in a **0–1 normalized** range. The LSTM pipeline does **not** apply `StandardScaler` on top of that — reported MAE/RMSE are **relative** errors on normalized consumption, not absolute kWh.

If LSTM-specific scaling is added later, apply `scaler.inverse_transform` to both `y_true` and `y_pred` before calling `evaluate_forecast`.

---

## What's Next

Per [Phase 3 Strategy](phase3-strategy.md):

1. ~~**Unified model comparison**~~ — **done:** [Forecast Model Comparison](forecast-model-comparison.md)
2. ~~Tutorial notebook~~ — **done:** [Forecasting Tutorial](forecasting-tutorial.md)
3. ~~Research write-up~~ — **done:** [Forecasting Research](forecasting-research.md)
4. Hyperparameter tuning (hidden size, epochs, learning rate)
5. Early stopping and model checkpointing

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Why sequence-level split:** Splitting the raw DataFrame before `create_sequences` would truncate windows at split edges incorrectly. Comparison and LSTM paths use `int(n * 0.7)` and `int(n * 0.85)` on `(X, y)` arrays — same fraction math as `time_series_split`.

**PyTorch → sklearn bridge:** `predict_lstm` must `.detach().cpu().numpy()` before `evaluate_forecast`; raw tensors are not accepted.

**Smoke test:**

```bash
python -m src.models.lstm_model   # forward-shape check (batch,) output
python scripts/verify_lstm_prep.py
python scripts/compare_forecasts.py
```

**Modularity:** Loads clean CSV only; does not retrain anomaly models unless you run `generate_clean_data.py` explicitly.

</details>

---

## References

- [LSTM Prep](lstm-prep.md) — 3D sequence generation
- [Forecast Model Comparison](forecast-model-comparison.md) — side-by-side ladder scoring
- [Forecasting Tutorial](forecasting-tutorial.md) — CMU educational notebook
- [XGBoost Forecasting](xgboost-forecasting.md) — tabular gradient-boosted baseline
- [Prophet Baseline](prophet-baseline.md) — statistical floor
- [Forecasting Baseline](forecasting-baseline.md) — gate, split, metrics
- [Feature Engineering](feature-engineering.md) — Phase 2 temporal columns
- [Phase 3 Strategy](phase3-strategy.md) — model ladder
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — EnergyLSTM, normalized forecast metrics
- [Getting Started](getting-started.md) — install and Phase 3 commands

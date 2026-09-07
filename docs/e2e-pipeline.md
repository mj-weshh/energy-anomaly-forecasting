# End-to-End Pipeline

Root **`main.py`** consolidates Phases 1–3 into one CLI: ingest and continuity check, features, Isolation Forest, in-memory cleaning, chronological split, a selected forecaster, test metrics, and a prediction CSV.

!!! success "Executive summary"

    - **One command:** `python main.py` runs ingest → continuity check → features → Isolation Forest → interpolate → chronological split → selected forecaster → metrics → prediction CSV.
    - **Active flags:** `--model` (`naive` / `prophet` / `xgboost` / `lstm`), `--epochs` (LSTM), `--output_path` (prediction CSV), optional `--save_clean_data`; default model is **naive**.
    - **Smoke-tested:** `python main.py --model naive` → 750 predictions; `--model xgboost` → 743 (lag warm-up).
    - **Terms:** [Glossary](glossary.md) — E2E pipeline / main.py, clean_pipeline_output, final_predictions, seasonal naive, supervised lag features.

**Entry point:** `main.py` (repository root)  
**Modules:** `src.data.ingest_data`, `src.features.build_features`, `src.models.train_anomaly_models`, `src.data.clean_data`, `src.data.make_forecast_dataset`, `src.models.train_forecast_models`, `src.models.lstm_model`, `src.models.evaluate_forecast`  
**Builds on:** [Getting Started](getting-started.md), [Anomaly Detection](anomaly-detection.md), [Clean Dataset](clean-data.md), [Forecasting Baseline](forecasting-baseline.md), [Forecast Model Comparison](forecast-model-comparison.md), [Forecasting Tutorial](forecasting-tutorial.md)

---

## Pipeline Steps

```text
python main.py [--data_path ...] [--model ...] [--epochs ...] [--save_clean_data] [--output_path ...]
  → load_smart_meter_data(data_path)     # Phase 1 — raw (5000, 7)
  → check_time_continuity(df)            # 30-minute cadence gate
  → build_all_features(df)               # Phase 2 early — (5000, 15)
  → detect_anomalies(..., isolation_forest)
  → interpolate_anomalies(df_feat, predictions)   # clean in memory
  → [optional] save data/processed/clean_pipeline_output.csv
  → time_series_split(df_clean)          # 70 / 15 / 15
  → run_selected_forecast(args.model)    # native prep + train/predict
  → MAE / RMSE / MAPE on length-matched y_true, y_pred
  → save Timestamp, y_true, y_pred CSV   # default: final_predictions.csv
```

```mermaid
flowchart LR
  cli[main.py_CLI] --> ingest[load_smart_meter_data]
  ingest --> continuity[check_time_continuity]
  continuity --> feats[build_all_features]
  feats --> detect[detect_anomalies_IF]
  detect --> clean[interpolate_anomalies]
  clean --> split[time_series_split]
  split --> route[run_selected_forecast]
  route --> eval[MAE_RMSE_MAPE]
  eval --> export[final_predictions_CSV]
```

`main.py` delegates to `src/` — it does **not** reimplement ingestion, detection, interpolation, forecast trainers, or metric helpers.

**Relationship to research scripts:** `main.py` trains **one** model per run and exports that run’s test predictions. For the four-model MAE/RMSE table and presentation PNG, use [`compare_forecasts.py`](forecast-model-comparison.md).

---

## CLI Reference

| Argument | Type | Default | Role |
|----------|------|---------|------|
| `--data_path` | `Path` | `Smart Meter Electricity Consumption Dataset/smart_meter_data.csv` | Path to the smart meter CSV |
| `--model` | choice | `naive` | Forecaster: `naive`, `prophet`, `xgboost`, `lstm` |
| `--epochs` | `int` | `20` | LSTM training epochs (ignored by other models) |
| `--save_clean_data` | flag | off | Save interpolated frame to `data/processed/clean_pipeline_output.csv` |
| `--output_path` | `Path` | `data/processed/final_predictions.csv` | Destination for test `Timestamp` / `y_true` / `y_pred` |

```bash
python main.py
python main.py --model naive
python main.py --model xgboost
python main.py --model lstm --epochs 30
python main.py --save_clean_data
python main.py --output_path path/to/my_predictions.csv
python main.py --data_path path/to/smart_meter_data.csv
```

On Windows:

```powershell
.venv\Scripts\activate
python main.py --model naive
```

---

## Logging

Configured at module load:

```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
```

### Example run (local, reproducible) — `--model naive`

```text
INFO: E2E pipeline starting (model=naive, epochs=20, data_path=..., save_clean_data=False, output_path=...)
INFO: Raw data loaded: shape=(5000, 7)
INFO: Checking 30-minute time-series continuity ...
INFO: Feature matrix ready: shape=(5000, 15) (47 rows with rolling-window warm-up NaNs; ...)
INFO: Anomalies detected: 248 of 4953 scored rows
INFO: Clean in-memory dataset ready: shape=(5000, 15), consumption_NaNs=0
INFO: Chronological train/val/test split (70/15/15) ...
INFO: train split: rows=3500, 2024-01-01 00:00:00 -> 2024-03-13 21:30:00
INFO: val split: rows=750, 2024-03-13 22:00:00 -> 2024-03-29 12:30:00
INFO: test split: rows=750, 2024-03-29 13:00:00 -> 2024-04-14 03:30:00
INFO: Training forecast model: naive (seasonal persistence) ...
INFO: Forecast complete for model=naive: prediction_length=750
INFO: Prediction preview (first 5 values): [0.353724, 0.281565, 0.312   , 0.717162, 0.556462]
INFO: Final Model Evaluation - MAE: 0.17114979670368508
INFO: Final Model Evaluation - RMSE: 0.21403447713731313
INFO: Final Model Evaluation - MAPE: ...
INFO: Saved final predictions (750 rows) to .../data/processed/final_predictions.csv
INFO: Pipeline execution completed successfully.
```

Anomaly count, preview values, and metric floats are from an **example** local run; re-runs may differ slightly. MAPE can look large on near-zero normalized consumption (same zero-safe helper as research scripts).

---

## Ingestion, Features, Detect, Clean

| Step | API | Expected result |
|------|-----|-----------------|
| Load | `load_smart_meter_data(data_path)` | Shape `(5000, 7)` |
| Continuity | `check_time_continuity(df)` | 30-minute cadence; no gaps / duplicates |
| Features | `build_all_features(df)` | Shape `(5000, 15)`; ~47 warm-up NaN rows |
| Detect | `detect_anomalies(..., model_type="isolation_forest")` | ~4953 scored rows |
| Clean | `interpolate_anomalies(df_feat, predictions)` | Shape `(5000, 15)`; 0 consumption NaNs |

Optional checkpoint: `python main.py --save_clean_data` → `data/processed/clean_pipeline_output.csv` (not the production `clean_smart_meter_data.csv` from `generate_clean_data.py`). See [Clean Dataset](clean-data.md).

---

## Chronological Split

```python
train_df, val_df, test_df = time_series_split(df_clean)  # 70 / 15 / 15
```

Example ranges on the default clean in-memory frame:

| Split | Rows | Timestamp range (example) |
|-------|------|---------------------------|
| Train | 3,500 | 2024-01-01 00:00 → 2024-03-13 21:30 |
| Val | 750 | 2024-03-13 22:00 → 2024-03-29 12:30 |
| Test | 750 | 2024-03-29 13:00 → 2024-04-14 03:30 |

Logged date ranges prove chronological order (no shuffle / no leakage).

---

## Model Routing

Implemented in `run_selected_forecast` — native prep matches [`compare_forecasts.py`](forecast-model-comparison.md) for naive / Prophet / XGBoost. **LSTM in `main.py` differs:** it splits chronological rows first, then windows sequences **within** each split so windows never cross train/val/test boundaries.

| `--model` | Prep | Train / predict | Test predictions (example) |
|-----------|------|-----------------|----------------------------|
| `naive` | Step-1 `train_df` / `test_df` | `naive_seasonal_forecast` (48-step) | **750** |
| `prophet` | Step-1 splits | `train_prophet_model` | **750** |
| `xgboost` | `create_supervised_lags(df_clean)` then **re-split** | `train_xgboost_model` + `predict` | **743** |
| `lstm` | **Split rows first**, then `create_sequences` on each split (7 features, `seq_length=24`) | `EnergyLSTM` + `train_lstm_model(epochs=args.epochs)` + `predict_lstm` | **747** |

XGBoost rebuilds splits after lag warm-up so incomplete early rows never enter the model. LSTM uses the already-split frames and windows inside each — same length outcome as the research sequence path, without leaking across boundaries.

`run_selected_forecast` returns length-matched `(timestamps, y_true, y_pred)` for the model’s native test window. The CLI logs `prediction_length` and a five-value preview.

---

## Evaluation & Export

After forecasting:

1. **Metrics** — `mean_absolute_error_forecast`, `root_mean_squared_error_forecast`, and `mean_absolute_percentage_error_forecast` from `src.models.evaluate_forecast`, logged as `Final Model Evaluation - MAE/RMSE/MAPE`.
2. **CSV export** — DataFrame with columns `Timestamp`, `y_true`, `y_pred` written to `--output_path` (default `data/processed/final_predictions.csv`). Parent directories are created as needed. `data/processed/` is gitignored.
3. **LSTM memory polish** — after inference, the LSTM path deletes the model, DataLoaders, and large arrays before return.
4. **Completion** — `Pipeline execution completed successfully.`

For side-by-side ladder metrics across all four models, keep using `compare_forecasts.py`.

---

## Related

- [Forecast Model Comparison](forecast-model-comparison.md) — four-model MAE/RMSE table and plot (`compare_forecasts.py`)
- [Forecasting Tutorial](forecasting-tutorial.md) · [`notebooks/04_forecasting_tutorial.ipynb`](../notebooks/04_forecasting_tutorial.ipynb)
- [Forecasting Research](forecasting-research.md) — findings write-up

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Modularity:** `main.py` imports public helpers only — ingest, `check_time_continuity`, features, `detect_anomalies`, `interpolate_anomalies`, `time_series_split`, forecast trainers, `EnergyLSTM`, and forecast metric helpers.

**CLI parsing:** `parse_args()` returns `data_path`, `model`, `epochs`, `save_clean_data`, and `output_path`.

**Constants:** Target `Electricity_Consumed`; seasonal periods 48; LSTM `seq_length=24`, `batch_size=32`; feature column lists match `compare_forecasts.py` where applicable.

**LSTM boundary rule:** Comment in `run_selected_forecast`: split rows first, then window within each split so sequences never cross train/val/test boundaries. Test timestamps are `test_df["Timestamp"].iloc[SEQ_LENGTH:]`.

**Smoke tests:**

```bash
python main.py --model naive
python main.py --model xgboost
```

**Commands:**

```bash
python main.py --help
python main.py --model prophet
python main.py --model lstm --epochs 20
python main.py --save_clean_data --model naive
python main.py --model naive --output_path data/processed/final_predictions.csv
```

</details>

---

## References

- [Getting Started](getting-started.md) — install and Phase 3 run commands
- [Feature Engineering](feature-engineering.md) — temporal, rolling, lags, sequences
- [Anomaly Detection](anomaly-detection.md) — Isolation Forest baseline
- [Clean Dataset](clean-data.md) — production clean artifact vs E2E checkpoint
- [Forecasting Baseline](forecasting-baseline.md) — split protocol and naive floor
- [Prophet Baseline](prophet-baseline.md) — statistical trainer
- [XGBoost Forecasting](xgboost-forecasting.md) — tabular lag model
- [LSTM Forecasting](lstm-forecasting.md) — recurrent trainer
- [Forecast Model Comparison](forecast-model-comparison.md) — four-model research aggregator
- [Forecasting Tutorial](forecasting-tutorial.md) — CMU educational notebook
- [Phase 3 Strategy](phase3-strategy.md) — model ladder and consolidation plan
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — E2E pipeline / main.py, final_predictions

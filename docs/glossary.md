# Glossary

Plain-English definitions for terms used across this project. Each entry includes a short **business read** and a **technical definition** for ML engineers.

!!! success "Executive summary"

    - **Purpose:** One place to decode jargon used in executive summaries and technical reports.
    - **How to use:** Skim the **Business** line for decisions; read **Technical** for implementation and reproducibility.
    - **Linked from:** Every docs page executive summary block points here for terms like F1, contamination, Jaccard, MAE / RMSE / MAPE, seasonal naive, Prophet, XGBoost, supervised lag features, PyTorch, and sliding window.

---

## Anomaly / Abnormal

**Business:** A smart-meter reading that looks unusual compared to normal patterns for that time of day and recent history — not necessarily a billing error, but worth investigating.

**Technical:** Rows labeled `Anomaly_Label = Abnormal` in the benchmark (5% of the dataset). Models are trained **without** this label; it is used only to score detection quality.

---

## Clean dataset / imputation

**Business:** A version of the consumption timeline where suspicious readings are replaced with sensible estimates so forecasting models see an unbroken 30-minute schedule.

**Technical:** `interpolate_anomalies` masks predicted anomalies on `Electricity_Consumed`, then time-interpolates gaps. Row count stays 5000; only consumption values at flagged intervals change.

---

## Contamination (Isolation Forest)

**Business:** How often the model is told to expect problems — a prior guess about the anomaly rate (~5% here).

**Technical:** sklearn `IsolationForest(contamination=...)` hyperparameter. Production default: `0.05`. Tuned enhanced config uses `0.03`.

---

## DBSCAN

**Business:** A second detection method that flags readings in sparse, lonely regions of the data — useful as a comparison, but weaker than Isolation Forest on this dataset.

**Technical:** Density-based clustering; points labeled `-1` (noise) map to Abnormal. Sensitive to `eps` and `min_samples` on multivariate features.

---

## Enhanced features (21 columns)

**Business:** A richer set of timing and consumption-change signals used in research tuning — not the default production pipeline.

**Technical:** `build_enhanced_anomaly_features`: legacy 15 columns plus cyclical time encodings (`hour_sin/cos`, `dow_sin/cos`) and derivatives (`consumption_diff`, `consumption_residual_24h`).

---

## eval_set (XGBoost)

**Business:** A way to watch validation performance while the model trains — without using the final test window for tuning decisions.

**Technical:** XGBoost `fit(..., eval_set=[(X_val, y_val)])` in `train_xgboost_model`. Validation loss is monitored during training; reported business metrics still come from the held-out **test** split via `evaluate_xgboost.py`.

---

## F1 score

**Business:** A single number that balances “did we catch real problems?” with “how many false alarms did we raise?” Higher is better; 1.0 is perfect.

**Technical:** Harmonic mean of precision and recall with **Abnormal = positive class**. Primary metric in `evaluate_anomaly_model`. A model predicting all Normal would score high on accuracy but F1 ≈ 0.

---

## Forecast chronological split (70/15/15)

**Business:** Cut the cleaned meter history into past (train), middle (validation), and future (test) — never mix them randomly.

**Technical:** `time_series_split` in `make_forecast_dataset.py` on the full 5,000-row clean CSV. Distinct from Phase 2 anomaly research **60/20/20** on eval rows. See [Forecasting Baseline](forecasting-baseline.md).

---

## False positive (FP)

**Business:** A false alarm — the model flagged a normal interval as a problem.

**Technical:** Confusion matrix cell: predicted Abnormal, actually Normal. Legacy IF has 165 FPs on the 991-row temporal test window; hours 00–01 concentrate most FPs.

---

## Isolation Forest (IF)

**Business:** The primary anomaly detector — it learns what “normal” multivariate patterns look like and flags readings that are quick to isolate (unusual).

**Technical:** sklearn `IsolationForest` on engineered features. Unsupervised; `Anomaly_Label` excluded from fit. Production cleaning uses default params; research uses tuned config in `anomaly_config.py`.

---

## Jaccard overlap

**Business:** How much two cleaning approaches agree on which intervals to fix — 0 means almost no overlap, 1 means identical choices.

**Technical:** `|A ∩ B| / |A ∪ B|` on sets of imputed row indices. Legacy vs enhanced clean artifacts: Jaccard ≈ 0.154.

---

## Legacy features (15 columns)

**Business:** The standard feature set used for production cleaning and teaching notebooks.

**Technical:** `build_all_features`: 7 original columns + `hour`, `day_of_week`, `month`, `is_weekend` + four rolling statistics over consumption.

---

## LSTM (prep stage)

**Business:** The deep-learning path that reads **stretches of recent history** (not just one row) before predicting the next interval — prep work shapes that history into tensors the network can consume.

**Technical:** Phase 3 Week 7 Day 3 implements **data prep only** via `create_sequences` and `verify_lstm_prep.py`. Default window: 24 steps (12 h). Full LSTM trainer not yet implemented. See [LSTM Prep](lstm-prep.md).

---

## MAE (Mean Absolute Error)

**Business:** On average, how far off are the forecasts? Easy to explain to management.

**Technical:** Mean of `|y_true - y_pred|` via `mean_absolute_error_forecast` / sklearn. Primary headline metric for Phase 3 model comparisons.

---

## MAPE (Mean Absolute Percentage Error)

**Business:** Relative forecast error as a percentage of the true value.

**Technical:** Mean of `|y_true - y_pred| / max(|y_true|, epsilon) × 100` with `epsilon=1e-8`. Can explode when true consumption is near zero on this normalized scale — prefer MAE/RMSE for headline comparisons. Implemented in `mean_absolute_percentage_error_forecast`.

---

## Prophet

**Business:** A statistical forecast model that learns daily and weekly patterns from timestamps — a stronger floor than “same time yesterday” without hand-built lag columns.

**Technical:** Facebook Prophet via `train_prophet_model` in `train_forecast_models.py`. Maps `Timestamp` → `ds` and `Electricity_Consumed` → `y`; fits on train only; returns `yhat` on the test horizon. Dependency: `prophet>=1.1.5`. See [Prophet Baseline](prophet-baseline.md).

---

## Precision

**Business:** When the model raises an alarm, how often is it right?

**Technical:** TP / (TP + FP). High precision = fewer false alarms.

---

## Profile (`--profile`)

**Business:** Which detection recipe to use when generating a clean CSV — default unchanged, or optional research variants.

**Technical:** `CleanProfile`: `legacy` (default IF on all rows), `legacy_threshold` (train/val threshold protocol), `enhanced` (tuned IF + 21 features). See [Clean Dataset — Research profiles](clean-data.md#research-profiles).

---

## PyTorch

**Business:** The deep-learning toolkit we use to build and train the LSTM forecaster — chosen for fine-grained control over model architecture.

**Technical:** `torch>=2.0.0` in `requirements.txt`. Day 3 prep converts NumPy sequence arrays to `torch.float32` tensors. Dependency for upcoming LSTM training code.

---

## Recall

**Business:** Of all real problems in the benchmark, what fraction did we catch?

**Technical:** TP / (TP + FN). High recall = fewer missed anomalies.

---

## RMSE (Root Mean Squared Error)

**Business:** Forecast error that punishes large misses more than MAE.

**Technical:** Square root of mean squared error. Uses `sklearn.metrics.root_mean_squared_error` (sklearn 1.6+); see `root_mean_squared_error_forecast`.

---

## Seasonal naive (48-step)

**Business:** The “same time yesterday” guess — tomorrow at 2:00 AM looks like today at 2:00 AM.

**Technical:** `naive_seasonal_forecast(..., seasonal_periods=48)` — at 30-minute resolution, 48 steps = 24 hours. Prediction at time `t` is the observed value at `t - 48` on `train || test_true` (not recursive). Phase 3 floor that advanced models must beat.

---

## Sliding window / sequence tensor

**Business:** A rolling slice of the recent timeline — like showing the model the last 12 hours of readings before asking it to guess the next half hour.

**Technical:** `create_sequences(data, seq_length=24)` returns `X` with shape `(num_samples, seq_length, n_features)` and `y` with shape `(num_samples, n_features)`. `num_samples = n_timesteps - seq_length`. See [LSTM Prep](lstm-prep.md).

---

## Supervised lag features

**Business:** Past consumption values placed on the same row as the value we want to predict — so tree models can “see” recent history without reading a clock.

**Technical:** `create_supervised_lags` adds `{target}_lag_1`, `_lag_2`, `_lag_48` (30 min, 1 h, 24 h at 30-minute resolution) and drops the first 48 incomplete rows. See [XGBoost Prep](xgboost-prep.md).

---

## Temporal split (60/20/20)

**Business:** Testing on future data the model has never seen — like grading on next month’s bills, not the same month used to tune settings. Used for **Phase 2 anomaly research**, not the Phase 3 forecast split.

**Technical:** Chronological split via `temporal_train_val_test_split`: train 2,971 / val 991 / test 991 eval rows. Hyperparameters and thresholds tuned on val; test F1 reported once. For forecasting, see **Forecast chronological split (70/15/15)**.

---

## Test split

**Business:** The final held-out time window used only to report honest performance numbers.

**Technical:** Last 20% of eval rows (991 rows). All fair-comparison F1s in [Anomaly Tuning Results](anomaly-tuning-results.md) use this slice.

---

## XGBoost (gradient boosting)

**Business:** An advanced machine-learning forecaster that combines many small decision trees — here fed with lags, weather, and calendar columns.

**Technical:** `XGBRegressor(n_estimators=100, learning_rate=0.1)` in `train_xgboost_model`. Trained on tabular features from [XGBoost Prep](xgboost-prep.md); scored via `evaluate_xgboost.py`. Dependency: `xgboost>=2.0.0`. See [XGBoost Forecasting](xgboost-forecasting.md).

---

## Threshold (score)

**Business:** A cutoff for how suspicious a reading must look before we flag it — tuned on validation data, not guessed.

**Technical:** Enhanced IF uses `score_threshold` from validation F1 sweep (`find_best_threshold` on isolation scores). Legacy threshold profile applies the same idea to legacy features.

---

## Weather ablation

**Business:** A check whether temperature, humidity, and wind columns help detection — on this dataset, removing them did not hurt (and slightly helped).

**Technical:** `tune_isolation_forest.py --drop-weather` drops weather columns from the feature matrix. Test F1 0.524 vs 0.460 with weather included.

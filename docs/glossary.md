# Glossary

Plain-English definitions for terms used across this project. Each entry includes a short **business read** and a **technical definition** for ML engineers.

!!! success "Executive summary"

    - **Purpose:** One place to decode jargon used in executive summaries and technical reports.
    - **How to use:** Skim the **Business** line for decisions; read **Technical** for implementation and reproducibility.
    - **Linked from:** Every docs page executive summary block points here for terms like F1, contamination, and Jaccard.

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

## F1 score

**Business:** A single number that balances “did we catch real problems?” with “how many false alarms did we raise?” Higher is better; 1.0 is perfect.

**Technical:** Harmonic mean of precision and recall with **Abnormal = positive class**. Primary metric in `evaluate_anomaly_model`. A model predicting all Normal would score high on accuracy but F1 ≈ 0.

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

## Precision

**Business:** When the model raises an alarm, how often is it right?

**Technical:** TP / (TP + FP). High precision = fewer false alarms.

---

## Profile (`--profile`)

**Business:** Which detection recipe to use when generating a clean CSV — default unchanged, or optional research variants.

**Technical:** `CleanProfile`: `legacy` (default IF on all rows), `legacy_threshold` (train/val threshold protocol), `enhanced` (tuned IF + 21 features). See [Clean Dataset — Research profiles](clean-data.md#research-profiles).

---

## Recall

**Business:** Of all real problems in the benchmark, what fraction did we catch?

**Technical:** TP / (TP + FN). High recall = fewer missed anomalies.

---

## Temporal split (60/20/20)

**Business:** Testing on future data the model has never seen — like grading on next month’s bills, not the same month used to tune settings.

**Technical:** Chronological split via `temporal_train_val_test_split`: train 2,971 / val 991 / test 991 eval rows. Hyperparameters and thresholds tuned on val; test F1 reported once.

---

## Test split

**Business:** The final held-out time window used only to report honest performance numbers.

**Technical:** Last 20% of eval rows (991 rows). All fair-comparison F1s in [Anomaly Tuning Results](anomaly-tuning-results.md) use this slice.

---

## Threshold (score)

**Business:** A cutoff for how suspicious a reading must look before we flag it — tuned on validation data, not guessed.

**Technical:** Enhanced IF uses `score_threshold` from validation F1 sweep (`find_best_threshold` on isolation scores). Legacy threshold profile applies the same idea to legacy features.

---

## Weather ablation

**Business:** A check whether temperature, humidity, and wind columns help detection — on this dataset, removing them did not hurt (and slightly helped).

**Technical:** `tune_isolation_forest.py --drop-weather` drops weather columns from the feature matrix. Test F1 0.524 vs 0.460 with weather included.

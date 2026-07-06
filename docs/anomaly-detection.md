# Anomaly Detection — Phase 2, Week 4

Working notes on the unsupervised anomaly detection engine. Week 3 gave us a 15-column feature matrix; Week 4 Day 1 wires that into an Isolation Forest baseline with proper imbalance-aware evaluation.

**Status:** Week 4 Day 1 complete — Isolation Forest baseline trained and benchmarked; DBSCAN next  
**Modules:** `src/models/evaluate_models.py`, `src/models/train_anomaly_models.py`  
**Builds on:** [Phase 2 Strategy](phase2-strategy.md), [Feature Engineering](feature-engineering.md)

---

## Why Isolation Forest First

The strategy doc describes what we need: a detector that flags readings that don't fit the local pattern, not a dumb threshold on raw consumption. Isolation Forest fits that story — it scores **multivariate weirdness** across all 12 engineered features (consumption, weather, temporal context, rolling memory) without needing balanced labels.

I'm starting here before DBSCAN because IF is fast, interpretable at the pipeline level, and gives us a baseline F1 to beat. DBSCAN comes next as a second opinion with different geometry assumptions.

---

## What's Implemented

### Evaluation — `evaluate_anomaly_model(y_true, y_pred)`

Imbalance-aware metrics for the benchmark comparison. Treats class **1** (Abnormal) as the positive class — no accuracy metric, because a model that always predicts Normal would hit 95%.

Returns precision, recall, F1, and a 2×2 confusion matrix.

### Training — `train_isolation_forest(df, contamination=0.05)`

Unsupervised fit on the engineered feature matrix. Critical design choices:

| Rule | Why |
|------|-----|
| **`Anomaly_Label` never used for training** | Labels are a benchmark only; the model must not see them during fit |
| **`prepare_feature_matrix` drops Timestamp + label + NaN rows** | 47 rolling warm-up rows removed; **4953 rows** remain for training |
| **12 numeric features** | Raw consumption + weather + temporal + rolling columns |
| **`contamination=0.05`** | Matches the ~5% Abnormal rate in the dataset; unsupervised prior, not supervised tuning |
| **Predictions mapped to 0/1** | sklearn returns `-1` (outlier) / `1` (inlier); we map to **1 = Abnormal, 0 = Normal** for evaluation |

### Pipeline helper — `build_all_features(df)`

One-call wrapper in `src/features/build_features.py`: temporal features, then rolling metrics.

### Verification — `scripts/test_isolation_forest.py`

End-to-end script: load → features → extract benchmark labels (aligned to the same `dropna` rows) → train → evaluate → print metrics.

---

## Baseline Results (Day 1)

Output from `python scripts/test_isolation_forest.py` on the real dataset:

```
Evaluation rows: 4953

Isolation Forest baseline metrics (Abnormal = positive class):
  Precision: 0.331
  Recall:    0.331
  F1:        0.331

Confusion matrix [[TN, FP], [FN, TP]]:
  TN=4539  FP= 166
  FN= 166  TP=  82
```

| Metric | Value | Notes |
|--------|-------|-------|
| Evaluation rows | 4,953 | 5,000 minus 47 rolling warm-up rows |
| Precision | 0.331 | Of predicted anomalies, ~33% match the benchmark |
| Recall | 0.331 | Of 248 benchmark Abnormal rows in eval set, ~82 caught |
| F1 | 0.331 | Harmonic mean; baseline, not tuned |

Honest read: the model catches roughly **one third** of benchmark anomalies at this default contamination. That's a starting point — not production-ready. Hyperparameter tuning and DBSCAN comparison are the next levers.

---

## How to Reproduce

```bash
pip install -r requirements.txt
python scripts/test_isolation_forest.py
```

Python API (same pipeline):

```python
from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data
from src.features.build_features import build_all_features
from src.models.train_anomaly_models import train_isolation_forest

df = build_all_features(load_smart_meter_data(find_dataset_csv(get_project_root())))
model, predictions = train_isolation_forest(df)
```

---

## What's Next

- **DBSCAN** — density-based second opinion; tune `eps` and `min_samples`
- **Hyperparameter tuning** — contamination, `n_estimators`, feature subsets
- **Experiment notebook** — traceability for CMU-Africa deliverables
- **Updated evaluation report** in docs once DBSCAN baseline exists

---

## References

- [Phase 2 Strategy](phase2-strategy.md) — why context-aware detection and imbalance-aware metrics matter
- [Feature Engineering](feature-engineering.md) — the 12 features fed into the model
- [Architecture](architecture.md) — where `src/models/` sits in the repo

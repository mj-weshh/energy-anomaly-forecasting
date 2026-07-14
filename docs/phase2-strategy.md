# Phase 2 Strategy — Anomaly Detection

Planning notes for the anomaly detection engine. Phase 1 is done — ingestion, schema checks, and EDA are all green. This page is what I'm using to align technical work with what we actually learned from the data.

**Status:** Week 4 complete — detectors benchmarked; clean dataset pipeline ready for Phase 3  
**Builds on:** [EDA Insights](eda-insights.md), [Verification Report](verification-report.md), [Feature Engineering](feature-engineering.md), [Anomaly Detection](anomaly-detection.md), [Clean Dataset](clean-data.md)

---

## What Phase 1 Gave Us (The Good)

I'm starting Phase 2 from a strong baseline. Week 1 proved the pipeline works: `python -m src.data.ingest_data` loads 5,000 rows at 30-minute intervals with **zero nulls**, correct dtypes, and a continuity check **PASS** — no gaps, no duplicates, no irregular spacing.

Week 2 EDA confirmed the dataset is ready for modeling without heavy cleanup:

- **Pre-normalized features (0–1).** `Electricity_Consumed`, weather columns, and `Avg_Past_Consumption` all sit in a tight 0–1 range. That saves us from building a scaling pipeline for Phase 2 — we can focus on context and multivariate patterns instead of fighting unit mismatches.
- **Clear time structure.** Consumption isn't flat across the day. Mean load peaks around **02:00**, with secondary bumps mid-morning. Day-of-week effects exist but are subtle (weekend mean slightly above weekday). Anomaly detection has to respect *when* a reading happened, not just *how high* it is.
- **Historical context matters.** `Avg_Past_Consumption` is the strongest linear relationship with current consumption (Pearson **r = +0.317**). Weather columns (`Temperature`, `Humidity`, `Wind_Speed`) show **negligible linear correlation** on this normalized slice — but they're still part of the multivariate picture; I won't drop them without testing.
- **Stable series.** The 104-day window has no obvious regime breaks or long gaps. Rolling averages smooth 30-minute noise without hiding structural problems.

Bottom line: I can spend Phase 2 on *detection logic*, not data janitor work.

---

## What We Have to Work Around (The Bad)

Clean data doesn't mean easy anomaly detection. A few constraints will shape every design choice:

### Normalized values, not physical units

Everything is scaled to ~0–1. We don't have raw kWh, °C, or m/s in this file. That's fine for a proof-of-concept and for comparing models on the same footing, but it means:

- Alarm thresholds won't translate directly to "bill impact" or facility ops language.
- Phase 3 forecasting metrics will be **relative** (error on normalized consumption), not absolute energy savings.

I'm noting this for stakeholders now so nobody expects dollar-denominated alerts in Phase 2.

### Severe label imbalance

The dataset ships with pre-assigned `Anomaly_Label` values. From EDA:

| Label | Count | Share |
|-------|-------|-------|
| Normal | 4,750 | 95.0% |
| Abnormal | 250 | 5.0% |

That's a **19 : 1** ratio — classic needle-in-a-haystack. Accuracy would be misleading (a model that always predicts "Normal" hits 95%). Phase 2 evaluation **must** use precision, recall, and F1 on the `Abnormal` class, even though our detectors train unsupervised.

Important: we treat `Anomaly_Label` as a **benchmark only**. Isolation Forest and DBSCAN won't see labels during training; labels are for post-hoc comparison.

### Context beats raw magnitude

Because load varies by hour and day, a single global threshold on `Electricity_Consumed` would fire constantly at peak hours and miss odd low-load events at night. Phase 2 has to be **context-aware** by design.

---

## The Strategy — Plain Language for Stakeholders

Here's the problem in non-technical terms:

**Energy use naturally moves up and down.** Hot afternoons, weekday mornings, and historical habits all change what "normal" looks like. A dumb alarm — "alert if consumption > X" — would cry wolf every afternoon and sleep through a real problem at 3 AM.

**What we need instead:** something like a smart security guard for the meter stream. It learns what the neighborhood usually does at each time of day and in each weather/history context. Then it flags readings that **don't fit the local pattern**, even if the raw number isn't extreme globally.

Concrete example:

- A **high** reading at **2 PM on a Tuesday** might be completely normal.
- The **same numeric reading at 3 AM on a cool Sunday** might be an anomaly — same spike, wrong context.

Our Phase 2 models (Isolation Forest and DBSCAN) fit this story: they score **multivariate weirdness** (consumption + history + weather + time context), not a single static cutoff.

---

## How EDA Drives Technical Execution

These aren't abstract goals — they're direct consequences of what we measured in Phase 1.

### Feature engineering

Because context matters more than raw level, I will:

- **Extract temporal features** from `Timestamp` — hour of day, day of week, weekend flag (same ideas as `add_temporal_features()` in Phase 1).
- **Build rolling statistics** on `Electricity_Consumed` — short-window mean and standard deviation so each interval is judged against its *local* recent behavior, not the global series.
- **Keep multivariate inputs together** — at minimum `Electricity_Consumed`, `Avg_Past_Consumption`, weather columns, and derived time/rolling features. Univariate thresholds alone won't cut it.

### Modeling approach

| Choice | Why |
|--------|-----|
| **Isolation Forest** | Handles high-dimensional, mostly-normal data well; isolates rare points without needing balanced labels |
| **DBSCAN** | Finds density-based outliers in feature space; useful as a second opinion with different geometry assumptions |
| **Unsupervised training** | Labels are too sparse and were generated externally — we benchmark against them, we don't train on them |
| **Imbalance-aware evaluation** | Precision / recall / F1 on `Abnormal`; confusion matrix; no accuracy-only reporting |

Both models are context-friendly when fed the engineered features above. Static z-score thresholds on raw consumption alone are explicitly **out of scope** for Phase 2.

### What Phase 2 delivered

- ~~Canonical module under `src/models/`~~ — **done** (`evaluate_models.py`, `train_anomaly_models.py`)
- ~~Isolation Forest baseline~~ — **done** (F1 = 0.331 on benchmark; see [Anomaly Detection](anomaly-detection.md))
- ~~DBSCAN detector and comparison~~ — **done** (best F1 = 0.125; IF wins on current grid)
- ~~Updated docs with results~~ — **done** (see [Anomaly Detection](anomaly-detection.md), [Clean Dataset](clean-data.md))
- ~~Clean interpolated dataset for Phase 3 forecasting~~ — **done** (IF default; see [Clean Dataset](clean-data.md))
- ~~Notebook for experiment traceability~~ — **done** — [`notebooks/03_anomaly_detection.ipynb`](../notebooks/03_anomaly_detection.ipynb)

### Explicitly not in Phase 2

- Forecasting (Phase 3)
- Deep learning detectors (keep the first pass interpretable and fast)
- Retraining on labels or SMOTE-style oversampling (stays unsupervised)

---

## Resolved decisions

These were open during planning; all are now closed:

1. **Rolling window length** — *resolved in Week 3:* both 3-hour and 24-hour windows implemented. See [Feature Engineering](feature-engineering.md).
2. **NaN warm-up rows** — *resolved in Week 4:* drop rows with incomplete rolling windows before training (4953 eval rows).
3. **DBSCAN hyperparameters** — *resolved:* legacy coarse grid F1 = 0.125; enhanced scaled grid test F1 = **0.297** at `eps=10`, `min_samples=10`, manhattan. See [Anomaly Tuning Results](anomaly-tuning-results.md).
4. **Ensemble vs pick-one** — *resolved for cleaning:* IF chosen as default cleaner (F1 0.331 vs DBSCAN 0.125). Ensemble union test F1 = 0.400 — below enhanced IF alone (0.460).
5. **Isolation Forest tuning** — *resolved:* enhanced IF test F1 = **0.460** vs legacy fair test F1 = **0.340** (production params) / **0.389** (val threshold). Production pipeline still uses legacy defaults. See [Anomaly Tuning Results](anomaly-tuning-results.md).

---

## References

- [EDA Insights](eda-insights.md) — load profiles, correlation heatmap, anomaly rate
- [Architecture](architecture.md) — ingestion gate and Phase 2 slot in the pipeline
- [Data Schema](data-schema.md) — column definitions
- [Anomaly Tuning Results](anomaly-tuning-results.md) — enhanced IF tuning, fair comparison, ensemble results

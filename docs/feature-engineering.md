# Feature Engineering — Phase 2, Week 3

Working notes on the feature engineering module. This is the first coding step of Phase 2 — turning raw timestamps into the temporal context our anomaly detectors need, plus rolling statistics that give the models short-term and daily memory.

**Status:** Temporal and rolling features complete; Week 4 anomaly detection done — see [Anomaly Detection](anomaly-detection.md)  
**Module:** `src/features/build_features.py`  
**Strategy background:** [Phase 2 Strategy](phase2-strategy.md)

---

## Why This Module Exists

Phase 1 EDA made one thing clear: consumption depends on *when* you look. Mean load peaks around **02:00**, and weekdays behave slightly differently from weekends. A detector that only sees the raw consumption number can't tell a normal afternoon spike from a suspicious 3 AM one.

Models don't read datetimes. They need that context as plain numeric columns. So I built a dedicated module, `src/features/`, as the canonical home for feature engineering — separate from ingestion (which validates data) and visualization (which plots it). Downstream scripts and notebooks import from here instead of re-deriving features ad hoc.

---

## What's Implemented — Temporal Features

`add_temporal_features(df)` takes the validated DataFrame from `src.data.ingest_data` and returns a **copy** with four integer columns added:

| Column | Range | Meaning |
|--------|-------|---------|
| `hour` | 0–23 | Hour of day |
| `day_of_week` | 0–6 | 0 = Monday … 6 = Sunday |
| `month` | 1–12 | Calendar month |
| `is_weekend` | 0 / 1 | 1 if Saturday or Sunday |

Design details worth knowing:

- **Copy semantics.** The function calls `df.copy()` first — your input DataFrame is never mutated. Cheap insurance against subtle bugs in notebooks.
- **Fail fast.** If `Timestamp` is missing, it raises `KeyError` immediately instead of producing garbage columns.
- **Integers, not booleans or categoricals.** `is_weekend` is 0/1 int, not `True`/`False`. Isolation Forest and DBSCAN work on numeric matrices; keeping everything integer-typed avoids dtype surprises at model time.

Usage:

```python
from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data
from src.features.build_features import add_temporal_features

df = add_temporal_features(load_smart_meter_data(find_dataset_csv(get_project_root())))
```

---

## What's Implemented — Rolling Metrics

`add_rolling_metrics(df)` gives the models **memory**. A high reading isn't suspicious if the last three hours were also high — the models need to see each interval against its recent neighborhood, not just in isolation. Four columns, all computed over `Electricity_Consumed`:

| Column | Window | Meaning |
|--------|--------|---------|
| `rolling_mean_3h` | 6 intervals (3 h) | Short-term local average |
| `rolling_std_3h` | 6 intervals (3 h) | Short-term local volatility |
| `rolling_mean_24h` | 48 intervals (24 h) | Daily baseline average |
| `rolling_std_24h` | 48 intervals (24 h) | Daily baseline volatility |

Why two windows: the **3-hour** window reacts fast and catches sudden deviations from the immediate trend (equipment kicking on, short spikes); the **24-hour** window holds the broader daily baseline. On the real data you can see the contrast — the 3h mean swings from 0.33 to 0.50 within a few intervals while the 24h mean stays anchored near 0.38.

Design details:

- **Chronological safety sort.** Rolling windows are garbage on out-of-order data, so the function does `df.sort_values("Timestamp").copy()` before any window math. Same copy semantics as the temporal function — the caller's frame is never touched.
- **NaN warm-up.** The first `window - 1` rows of each metric are NaN until the window fills (5 rows for 3h, 47 for 24h). That's expected pandas behavior. Week 4 drops those rows before training — see [Anomaly Detection](anomaly-detection.md).
- **Fail fast.** Missing `Timestamp` or `Electricity_Consumed` raises `KeyError` immediately.

---

## What's Implemented — Full Pipeline

`build_all_features(df)` chains temporal then rolling features in one call — the entry point downstream model code uses:

```python
from src.features.build_features import build_all_features

df = build_all_features(df)  # (5000, 15)
```

Same copy and sort semantics as the individual functions; no extra logic beyond calling both in order.

---

`scripts/verify_features.py` (which replaced the earlier `verify_temporal.py`) loads the real dataset, applies both functions, and sanity-checks everything:

```bash
python scripts/verify_features.py
```

Output from the actual run:

```
Shape after feature engineering: (5000, 15)

Temporal features (head):
          Timestamp  hour  day_of_week  month  is_weekend
2024-01-01 00:00:00     0            0      1           0
2024-01-01 00:30:00     0            0      1           0
2024-01-01 01:00:00     1            0      1           0
2024-01-01 01:30:00     1            0      1           0
2024-01-01 02:00:00     2            0      1           0

Rolling metrics (tail — windows fully filled):
          Timestamp  rolling_mean_3h  rolling_std_3h  rolling_mean_24h  rolling_std_24h
2024-04-14 01:30:00         0.328597        0.169606          0.377784         0.169480
2024-04-14 02:00:00         0.406674        0.092049          0.381593         0.169995
2024-04-14 02:30:00         0.474165        0.221625          0.390893         0.185203
2024-04-14 03:00:00         0.499800        0.213968          0.392865         0.185968
2024-04-14 03:30:00         0.460499        0.240648          0.385569         0.185183

PASS — temporal and rolling feature columns present with valid values.
```

7 original columns + 4 temporal + 4 rolling = 15. The script asserts every column exists, temporal values stay in range, the rolling tail is fully populated, and the warm-up rows are NaN exactly where they should be.

---

## Why Not Reuse the EDA Helper?

`src/visualization/visualize.py` already has an `add_temporal_features` for plotting — it adds a `day_name` ordered categorical so charts get readable axis labels. I deliberately did **not** reuse it for modeling:

- The plotting version returns categoricals and booleans, which are convenient for seaborn but noise for a model feature matrix.
- Keeping model features in `src/features/` means the visualization module can evolve for charts without silently changing model inputs.

Same idea, two audiences. When the two drift, the `src/features/` version is the one models trust.

---

## What's Next

- **Week 4 complete** — IF + DBSCAN baselines; clean dataset for Phase 3. See [Anomaly Detection](anomaly-detection.md) and [Clean Dataset](clean-data.md).

---

## References

- [Phase 2 Strategy](phase2-strategy.md) — why context-aware features matter
- [EDA Insights](eda-insights.md) — the 02:00 peak and weekday/weekend findings
- [Anomaly Detection](anomaly-detection.md) — Week 4 IF + DBSCAN baselines and model comparison
- [Clean Dataset](clean-data.md) — Week 4 Day 3 imputation pipeline for Phase 3
- [Architecture](architecture.md) — where `src/features/` sits in the repo

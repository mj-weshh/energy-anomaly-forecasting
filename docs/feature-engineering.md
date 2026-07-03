# Feature Engineering — Phase 2, Week 3

Working notes on the feature engineering module. This is the first coding step of Phase 2 — turning raw timestamps into the temporal context our anomaly detectors need.

**Status:** In progress (temporal features done; rolling statistics next)  
**Module:** `src/features/build_features.py`  
**Strategy background:** [Phase 2 Strategy](phase2-strategy.md)

---

## Why This Module Exists

Phase 1 EDA made one thing clear: consumption depends on *when* you look. Mean load peaks around **02:00**, and weekdays behave slightly differently from weekends. A detector that only sees the raw consumption number can't tell a normal afternoon spike from a suspicious 3 AM one.

Models don't read datetimes. They need that context as plain numeric columns. So I built a dedicated module, `src/features/`, as the canonical home for feature engineering — separate from ingestion (which validates data) and visualization (which plots it). Downstream scripts and notebooks import from here instead of re-deriving features ad hoc.

---

## What's Implemented

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

## How to Verify

There's a small script that loads the real dataset, applies the function, and sanity-checks the output:

```bash
python scripts/verify_temporal.py
```

Output from the actual run:

```
Shape after feature extraction: (5000, 11)

          Timestamp  hour  day_of_week  month  is_weekend
2024-01-01 00:00:00     0            0      1           0
2024-01-01 00:30:00     0            0      1           0
2024-01-01 01:00:00     1            0      1           0
2024-01-01 01:30:00     1            0      1           0
2024-01-01 02:00:00     2            0      1           0

PASS — all temporal columns present with valid value ranges.
```

7 original columns + 4 temporal features = 11. The script asserts that every column exists and that values stay in their valid ranges (`hour` 0–23, `day_of_week` 0–6, `month` 1–12, `is_weekend` in {0, 1}).

---

## Why Not Reuse the EDA Helper?

`src/visualization/visualize.py` already has an `add_temporal_features` for plotting — it adds a `day_name` ordered categorical so charts get readable axis labels. I deliberately did **not** reuse it for modeling:

- The plotting version returns categoricals and booleans, which are convenient for seaborn but noise for a model feature matrix.
- Keeping model features in `src/features/` means the visualization module can evolve for charts without silently changing model inputs.

Same idea, two audiences. When the two drift, the `src/features/` version is the one models trust.

---

## What's Next

- **Rolling statistics** — local mean and standard deviation on `Electricity_Consumed` so each interval is judged against its recent neighborhood, not the global series.
- **Week 4** — feed the assembled feature matrix into Isolation Forest and DBSCAN, evaluated against the 250-row `Abnormal` benchmark (see [Phase 2 Strategy](phase2-strategy.md)).

---

## References

- [Phase 2 Strategy](phase2-strategy.md) — why context-aware features matter
- [EDA Insights](eda-insights.md) — the 02:00 peak and weekday/weekend findings
- [Architecture](architecture.md) — where `src/features/` sits in the repo

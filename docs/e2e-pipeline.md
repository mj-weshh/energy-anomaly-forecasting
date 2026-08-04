# E2E Pipeline — Phase 3, Week 8 (Day 2)

Working notes for the root **`main.py`** entry point that consolidates the Phase 1–3 workflow into a single CLI command. Day 2 scaffolds argparse, centralized logging, and wires ingestion plus early feature engineering.

<div class="admonition success" markdown="1">
<p class="admonition-title">Executive summary</p>

- **One command:** `python main.py` is the user-facing entry point at the repository root.
- **Day 2 scope:** CLI flags + INFO logging + Phase 1 ingestion + Phase 2 temporal/rolling features via existing `src/` APIs.
- **Verified shapes:** Raw `(5000, 7)` → engineered `(5000, 15)` with **47** rolling warm-up NaN rows (timeline preserved; no rows dropped).
- **Reserved flags:** `--model` and `--epochs` are parsed today but not yet used for training — anomaly detection, cleaning, and forecasting wire in later E2E days.
- **Terms:** [Glossary](glossary.md) — E2E pipeline / main.py, rolling warm-up, feature engineering.

</div>

**Status:** Week 8 Day 2 complete — CLI scaffold, logging, ingestion, and feature engineering  
**Entry point:** `main.py` (repository root)  
**Modules:** `src.data.ingest_data`, `src.features.build_features`  
**Builds on:** [Getting Started](getting-started.md), [Feature Engineering](feature-engineering.md), [Forecast Model Comparison](forecast-model-comparison.md)

---

## End-to-End Pipeline (current vs planned)

```text
python main.py [--data_path ...] [--model ...] [--epochs ...]
  → load_smart_meter_data(data_path)     # Phase 1 — raw (5000, 7)
  → build_all_features(df)               # Phase 2 early — (5000, 15)
  → [planned] anomaly detection + clean artifact
  → [planned] forecast with --model / --epochs
```

```mermaid
flowchart LR
  cli[main.py_CLI] --> ingest[load_smart_meter_data]
  ingest --> feats[build_all_features]
  feats --> later[anomaly_clean_forecast_planned]
```

`main.py` delegates to `src/` — it does **not** reimplement ingestion or feature logic.

---

## CLI Reference

| Argument | Type | Default | Role |
|----------|------|---------|------|
| `--data_path` | `Path` | `Smart Meter Electricity Consumption Dataset/smart_meter_data.csv` | Path to the smart meter CSV |
| `--model` | choice | `naive` | Reserved: `naive`, `prophet`, `xgboost`, `lstm` |
| `--epochs` | `int` | `20` | Reserved: LSTM training epochs |

```bash
python main.py
python main.py --model lstm --epochs 30
python main.py --data_path path/to/smart_meter_data.csv
```

On Windows:

```powershell
.venv\Scripts\activate
python main.py
```

---

## Logging

Configured at module load:

```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
```

Pipeline messages use `logger.info(...)` rather than `print()`.

### Example run (local, reproducible)

```text
INFO: E2E pipeline starting (model=naive, epochs=20, data_path=...)
INFO: Starting data ingestion from ... ...
INFO: Raw data loaded: shape=(5000, 7)
INFO: Building temporal and rolling features ...
INFO: Feature matrix ready: shape=(5000, 15) (47 rows with rolling-window warm-up NaNs; row count preserved - no rows dropped)
```

---

## Ingestion and Feature Engineering

| Step | API | Expected result |
|------|-----|-----------------|
| Load | `load_smart_meter_data(data_path)` | Shape `(5000, 7)`, parsed `Timestamp` |
| Features | `build_all_features(df)` | Shape `(5000, 15)` — temporal + rolling columns |

**Warm-up NaNs:** Rolling windows leave the first incomplete rows as NaN (**47** rows on the default dataset for the 24-hour window). Day 2 logs this count and **does not drop** rows — same continuity rule as Phase 2 feature verification.

Individual scripts (`python -m src.data.ingest_data`, `scripts/verify_features.py`) remain valid for focused checks. `main.py` is the consolidating entry path.

---

## What's Next

Per [Phase 3 Strategy](phase3-strategy.md) E2E consolidation:

1. Wire anomaly detection and clean-data generation into `main.py`
2. Wire chronological split and `--model` forecasting (naive / Prophet / XGBoost / LSTM)
3. Research write-up and tutorial notebook (`forecasting-research.md`, `04_forecasting_tutorial.ipynb`)

---

<details class="info" markdown="1">
<summary>Technical deep dive</summary>

**Modularity:** `main.py` imports only public helpers from `src.data.ingest_data` and `src.features.build_features`. No duplicated CSV parsing or rolling math.

**CLI parsing:** `parse_args()` returns an `argparse.Namespace` with `data_path`, `model`, and `epochs`. Model/epochs are logged at startup for forward compatibility.

**Warm-up count:** `int(df_feat.isna().any(axis=1).sum())` after `build_all_features` — typically 47 on the 5,000-row Kaggle CSV (first 47 rows incomplete for the 48-step / 24h rolling window).

**Commands:**

```bash
python main.py
python main.py --help
```

</details>

---

## References

- [Getting Started](getting-started.md) — install and Phase 3 run commands
- [Feature Engineering](feature-engineering.md) — temporal and rolling columns
- [Forecast Model Comparison](forecast-model-comparison.md) — research ladder aggregator (separate from E2E CLI)
- [Phase 3 Strategy](phase3-strategy.md) — model ladder and consolidation plan
- [Architecture](architecture.md) — repository layout
- [Glossary](glossary.md) — E2E pipeline / main.py

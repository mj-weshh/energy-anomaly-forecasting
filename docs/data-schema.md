# Data Schema

Reference documentation for the Smart Meter Electricity Consumption Dataset used in this project.

!!! success "Executive summary"

    - **What you get:** 5,000 half-hour readings from Jan–Apr 2024 — electricity use, weather, past-average context, and an expert anomaly label.
    - **Business columns:** `Electricity_Consumed` (usage), `Avg_Past_Consumption` (historical baseline), weather for context.
    - **Quality bar:** No nulls; timestamps every 30 minutes; labels are `Normal` or `Abnormal`.
    - **Terms:** [Glossary](glossary.md) — anomaly label, continuity.

**Source:** [Kaggle — Smart Meter Electricity Consumption Dataset](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset)  
**Local filename:** `smart_meter_data.csv`  
**Verified on:** Phase 1, Week 1 ingestion run

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 7 |
| Sampling frequency | 30 minutes |
| Start timestamp | `2024-01-01 00:00:00` |
| End timestamp | `2024-04-14 03:30:00` |
| Missing values | 0 (all columns) |
| Continuity | PASS — no gaps, duplicates, or irregular intervals |

---

## Data Dictionary

| Column | Dtype | Nullable | Description | Example |
|--------|-------|----------|-------------|---------|
| `Timestamp` | `datetime64[ns]` | No | Start of each 30-minute measurement interval | `2024-01-01 00:00:00` |
| `Electricity_Consumed` | `float64` | No | Normalized electricity consumption for the interval | `0.458` |
| `Temperature` | `float64` | No | Normalized ambient temperature | `0.470` |
| `Humidity` | `float64` | No | Normalized relative humidity | `0.396` |
| `Wind_Speed` | `float64` | No | Normalized wind speed | `0.445` |
| `Avg_Past_Consumption` | `float64` | No | Normalized rolling average of historical consumption | `0.692` |
| `Anomaly_Label` | `object` (string) | No | Pre-assigned class label: `Normal` or `Abnormal` | `Normal` |

### Notes on numeric features

All numeric columns except `Timestamp` appear **normalized** (values roughly in the 0–1 range). Raw physical units (kWh, °C, etc.) are not present in the file. Downstream EDA and modeling should treat these as scaled features unless denormalization metadata becomes available.

### Notes on labels

`Anomaly_Label` is a binary categorical field:

- `Normal` — majority class (95.0%, 4,750 rows)
- `Abnormal` — minority class (5.0%, 250 rows)

Baseline label distribution and figures: [EDA Insights](eda-insights.md).

Phase 2 unsupervised methods will not rely on this label for training; it is retained for evaluation and supervised baselines.

---

## File Discovery

The ingestion module locates the CSV dynamically. Search order (first match wins):

1. `data/raw/*.csv`
2. `Smart Meter Electricity Consumption Dataset/*.csv`
3. Any `**/smart_meter_data.csv` under the project root

Implementation: `src/data/ingest_data.py` — `find_dataset_csv()`

---

## Timestamp Format

| Property | Value |
|----------|-------|
| Column name | `Timestamp` |
| String format | `%Y-%m-%d %H:%M:%S` |
| Parsed dtype | `datetime64[ns]` |
| Timezone | None (naive datetime) |
| Expected delta | 30 minutes between consecutive rows |

Parsing is performed in `load_smart_meter_data()`:

```python
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
```

---

## Expected Schema Contract

Any CSV accepted by this project must satisfy:

- Exactly 7 columns with the names listed above
- No null values in any column
- Monotonically increasing timestamps at 30-minute intervals
- `Anomaly_Label` values limited to `Normal` and `Abnormal`

Violations are reported by `print_schema_summary()` and `check_time_continuity()` in the ingestion module.

??? info "Technical deep dive"

    **dtypes:** `Timestamp` → `datetime64[ns]`; numerics → `float64`; `Anomaly_Label` → `object` (`Normal` / `Abnormal`).

    **Ranges:** Normalized features in ~0–1; `Anomaly_Label` imbalanced (~5% `Abnormal`).

    **Contract enforcement:** `src/data/ingest_data.py` — `load_smart_meter_data`, `print_schema_summary`, `check_time_continuity`.

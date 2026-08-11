# Forecasting Research — Phase 3 Findings

Research notes on what Phase 3 actually taught us about forecasting half-hour smart-meter load. Phase 2 left a continuous cleaned timeline; this page asks how predictable that series is once anomalies are repaired — and which model on our ladder earns the headline.

!!! success "Executive summary"

    - **Question:** On the production clean CSV, how well can we forecast the next intervals — and which model wins under our default protocol?
    - **Protocol:** Chronological train / val / test; score native test windows with MAE and RMSE via `compare_forecasts.py`.
    - **Headline:** **Prophet** leads MAE and RMSE on the Week 8 default run; LSTM is close; all advanced models beat the naive seasonal floor.
    - **Terms:** [Glossary](glossary.md) — MAE, RMSE, seasonal naive, forecast model comparison.

**Status:** Week 9 Days 3–4 — research write-up in progress  
**Builds on:** [Forecast Model Comparison](forecast-model-comparison.md), [Forecasting Baseline](forecasting-baseline.md), [Phase 3 Strategy](phase3-strategy.md), [Clean Dataset](clean-data.md)

---

## What Phase 2 Handed Off

I'm not starting forecasting from a broken timeline. Phase 2's production artifact — `data/processed/clean_smart_meter_data.csv` — keeps all 5,000 half-hour rows, fills flagged consumption with time interpolation, and leaves the meter stream continuous. That matters: a forecaster that trains on gaps or deleted anomaly rows is answering a different question than “what does demand look like next?”

Phase 3's job was narrower than inventing new cleaning recipes. It was: **given this cleaned series, how predictable is load**, and does a model ladder (naive → Prophet → XGBoost → LSTM) beat a honest seasonal floor without leaking the future into training?

---

## Model Ladder Results

We scored each model on its **native** chronological test pipeline — the same paths wired in [`scripts/compare_forecasts.py`](../scripts/compare_forecasts.py) and documented in [Forecast Model Comparison](forecast-model-comparison.md). Re-run locally with:

```bash
python scripts/compare_forecasts.py
```

Example test metrics from the Week 8 default hyperparameter run:

| Model | MAE | RMSE | Test rows |
|-------|-----|------|-----------|
| Naive | 0.171150 | 0.214034 | 750 |
| Prophet | **0.121071** | **0.148670** | 750 |
| XGBoost | 0.125274 | 0.153876 | 743 |
| LSTM | 0.122156 | 0.151200 | 747 |

Lower MAE and RMSE are better. Test row counts differ because XGBoost and LSTM re-split after lag / sequence warm-up — so this is **not** one shared timestamp panel; it is a fair per-model native evaluation.

### Who won — and why I'm not surprised

**Prophet wins this run** on both MAE and RMSE.

That result is sharper than “deep learning always wins.” A few observations:

- **The naive floor is real.** Seasonal persistence (repeat the value from 48 steps ago / 24 hours) is a hard baseline on a series with clear daily structure. Every advanced model clears it — if something couldn't beat ~0.171 MAE, I wouldn't trust it.
- **Prophet fits this series' shape.** The load profile is normalized, fairly smooth, and diurnal. Prophet's trend + seasonality story matches that structure without needing a large tabular feature matrix. Leading both MAE and RMSE under default settings is the honest headline.
- **LSTM is close.** At ~0.122 MAE it beats XGBoost and sits near Prophet — competitive for a small CPU-trained network, not a runaway win.
- **XGBoost is solid but third.** Lag + weather + calendar features get it past naive, yet on this default `n_estimators=100` / `learning_rate=0.1` setup it trails Prophet and LSTM on average error. Trees react to lag structure; they don't automatically invent Prophet-style global seasonality.

I'm treating Prophet as the **best default forecaster for this artifact and protocol**, not as proof that statistical models always dominate tree or sequence models. Hyperparameter search, longer horizons, or a different meter could reorder the ladder. For Week 8 as shipped: Prophet is the winner.

---

## References

- [Forecast Model Comparison](forecast-model-comparison.md) — reproducible ladder script and plot
- [Forecasting Baseline](forecasting-baseline.md) — chronological split and naive floor
- [Prophet Baseline](prophet-baseline.md) — statistical trainer
- [XGBoost Forecasting](xgboost-forecasting.md) — tabular lag model
- [LSTM Forecasting](lstm-forecasting.md) — sequence model
- [Phase 3 Strategy](phase3-strategy.md) — model ladder plan
- [Glossary](glossary.md) — MAE, RMSE, seasonal naive

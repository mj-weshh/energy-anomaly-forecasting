# Phase 2 Progress Report

**Project:** Energy Anomaly Forecasting  
**Program:** CMU-Africa Techskills Internship — Wawtex Solutions Limited  
**Prepared by:** Waweru Muhura 
**Prepared for:** Raymond Namusoso 
**Report date:** 22 July 2026  
**Scope:** Phase 2 complete; Phase 3 outlined  

---

## 1. Purpose of this report

I am writing this report so you can review what I delivered in Phase 2, what the results mean for the project, and what I will do next in Phase 3—without needing a machine-learning background.

Phase 2 had one practical goal: **spot unusual electricity readings on a smart meter, then repair them carefully** so we keep a complete half-hour timeline ready for demand forecasting. I treated the meter like a diary that writes an entry every thirty minutes. My job was to find pages that look wrong, fix the smudges, and never tear pages out—because forecasting needs every entry in order.

All of this work uses only a **public open dataset** (the Kaggle Smart Meter Electricity Consumption Dataset). No proprietary customer systems or private meter feeds are involved, which keeps us aligned with NDA discipline while still building real, reusable engineering.

---

## 2. Where we stood entering Phase 2

Phase 1 left me with a trustworthy foundation:

- **5,000** half-hour readings over roughly a hundred days  
- **Zero** missing values  
- Perfect spacing—no gaps, no duplicates, no irregular intervals  

Exploration also showed patterns that shaped Phase 2. Average use **peaks around 2 a.m.**, so “high” is not automatically “wrong.” Weather (temperature, humidity, wind) barely tracks consumption on this normalized data, while **past average use** is the strongest simple clue that history matters. Only about **1 in 20** readings are labeled abnormal—so problems are rare, and a system that always says “everything is fine” would look accurate while missing the point.

That set up Phase 2 clearly: I needed context-aware detection, honest scoring when issues are rare, and a cleaning method that preserves the full timeline.

---

## 3. Making the meter “context-aware”

A fixed “if consumption is above X, flag it” rule would fail here. The same number at 2 a.m. can be normal for this building and suspicious at midday. A good utility operator already thinks that way; I built that judgment into the data the models see.

I engineered context in three layers:

| Layer | What I added | Why it matters |
|-------|----------------|----------------|
| **Time of day and calendar** | Hour, day of week, month, weekend flag | Distinguishes night peaks from daytime spikes |
| **Recent memory** | 3-hour and 24-hour rolling averages and volatility | Asks “is this odd *relative to the last few hours / last day*?” |
| **Stronger research context** | Smooth cyclical time encoding; step-to-step change; difference from the daily baseline | Gives the research models a richer picture of change and rhythm |

The production path uses the first two layers (**15** columns total including the original fields). The research path builds on that to **21** columns. Rolling windows need a short warm-up at the start of the series; I exclude those incomplete rows before scoring, so models judge **4,953** complete intervals—not incomplete ones.

In short: I taught the system *when* a reading happened and *how it sits against recent history*, not just how large the number is.

---

## 4. Finding unusual consumption

I built and compared two automatic “watchdogs” that look for odd readings without being told the answer key while they learn:

| Approach | Plain-language role | Result on our baseline test |
|----------|---------------------|-----------------------------|
| **Isolation Forest** | Separates unusual points in a multi-factor picture of use and context | Balanced catch-and-correct score of about **0.33** (see note below) |
| **DBSCAN** | Groups “normal neighborhoods” and treats loners as suspicious | Weaker on this data—about **0.13**; tended to cry wolf too often |

**How I graded them.** The dataset includes expert labels (`Normal` / `Abnormal`), but I **hid those labels during training**. I only used them afterward to score quality—like grading a new field inspector against known problem cases after the shift, not handing them the answer sheet.

Because abnormal cases are rare (~5%), ordinary “percent correct” is misleading. I used a balanced score that asks two questions at once: *of the intervals we flagged, how many were real problems?* and *of the real problems, how many did we catch?* That combined balance is what the **0.33** and **0.13** numbers represent (technically an F1 score on the abnormal class). Higher is better; **1.0** would be perfect catch-and-correct balance.

**Production choice.** Isolation Forest was the clearer winner for the default cleaning path. It flagged on the order of **~250** intervals—close to the known ~5% problem rate—rather than over-flagging hundreds of extras. That conservatism is intentional for a first clean pass before forecasting.

I also documented the full detect → score → clean workflow in an educational notebook so the path is reproducible for review and teaching.

---

## 5. Research and fine-tuning

Beyond the production baseline, I ran a structured research program so we would know whether a stronger recipe is worth adopting later—without quietly changing the file Phase 3 will rely on.

### Fair testing on “future” data

I evaluated improved setups on a **chronological** split: train on earlier periods, tune on a middle window, and score only on a later held-out window. That is closer to real operations than scoring on the same days the model has already seen—more like grading a forecast on next month’s bills than on last month’s.

### What I tuned

On the research path I:

- Expanded context to the **21-column** feature set  
- Fitted scaling **only on the training window** so later periods are judged fairly (important for distance-based methods)  
- Adjusted Isolation Forest settings (how often it expects problems, how many trees, and the score threshold chosen on validation)  
- Retuned the second watchdog and tried combining both (union / intersection strategies)  
- Ran a **weather drop** experiment to see if temperature, humidity, and wind were carrying the gains  
- Broke down errors by hour and by weekday vs weekend  
- Generated alternate cleaned files under research “profiles” and measured how much they disagree with the default clean file  

### Head-to-head results (held-out future window)

| Approach | Balanced score on future test window | What it means |
|----------|--------------------------------------|---------------|
| Default Isolation Forest (production-style settings) | **~0.34** | Solid baseline on unseen time |
| Same recipe, only the decision threshold tuned on validation | **~0.39** | Better grading of the same style of model |
| **Enhanced, fully tuned Isolation Forest** | **~0.46** | Best single model—clear uplift |
| Enhanced second watchdog (DBSCAN) | **~0.30** | Improved vs its old baseline, still behind Isolation Forest |
| Combining both watchdogs (union) | **~0.40** | Helpful, but still below enhanced Isolation Forest alone |

Relative to the production-style score on the same future window (~0.34), the enhanced Isolation Forest is about **+0.12** on that balanced scale—materially better catch-and-correct performance without pretending the problem is solved.

### What else the research told us

**Weather is not the hero here.** Dropping weather columns from the enhanced model slightly *improved* the future-window score (about **0.52** vs **0.46** with weather). That matches Phase 1: on this dataset, past use and time context matter more than outdoor conditions for spotting odd readings.

**False alarms cluster early.** Under the legacy watchdog on the test window, mistaken flags concentrated in the **midnight–1 a.m.** hours—not at the 2 a.m. peak itself. Peak load is not the same thing as “most false alarms.” That tells us where to look if we tighten the system later.

**Weekends and early morning remain harder.** Even after tuning, weekend catch-rate sits a bit below the overall enhanced level, and several early-morning and some evening hours stay below the global average. The research map shows where the next gains would need to come from.

**Different cleaning recipes edit different half-hours.** I compared three cleaning profiles side by side:

| Cleaning recipe | Roughly how many half-hours get repaired | Overlap with default |
|-----------------|------------------------------------------|----------------------|
| Default (legacy Isolation Forest) | **~248** | — |
| Legacy + validation threshold | **~178** | Moderate disagreement |
| Enhanced research recipe | **~51** | Low overlap with default (~15% agreement on *which* intervals are edited) |

So “better detection on paper” is not a free upgrade to the cleaned history. The enhanced recipe repairs **far fewer** intervals and largely **different** ones. Switching without review would change the story Phase 3 learns from.

---

## 6. What this means for the project

From an executive point of view, Phase 2 leaves us in a strong, controlled position:

1. **We have a stable production path.** The default clean file still uses the conservative Isolation Forest recipe. Phase 3 can start on a known, documented artifact without surprise changes.

2. **We have measured upside.** Research shows a clearly stronger detector on future-held-out data (~0.46 vs ~0.34 on the same fair test). That is evidence, not guesswork.

3. **We know the trade-offs.** Adopting the research cleaner would rewrite fewer intervals—but not the same ones. That is a leadership decision, not something that should be flipped silently in the middle of forecasting work.

4. **We know what not to chase.** Weather is optional on this benchmark. Combining both watchdogs did not beat the best single Isolation Forest. Early morning and weekends are the remaining hard segments.

**Decision gate for your review:** Keep the default cleaned history for Phase 3 consistency (my recommendation to proceed now), and treat the enhanced profile as an optional A/B later if we want fewer, more precise repairs after a deliberate review.

---

## 7. Cleaning for forecasting

Detection alone is not enough for the next phase. Forecasting models need an unbroken half-hour schedule. If I simply deleted suspicious rows, I would create holes—like ripping pages from the diary.

So I built a cleaning pipeline that:

1. **Flags** suspicious consumption with Isolation Forest  
2. **Blanks** only those consumption values (other fields stay put)  
3. **Fills** the blanks with a time-based smooth estimate from neighboring readings  
4. **Writes** a complete file: still **5,000** rows, zero missing consumption after the fill  

Under the default recipe, that means about **248** half-hours repaired—and a continuous series ready for forecasting. Research profiles write separate files so we can compare recipes without overwriting the baseline.

---

## 8. Phase 3 — what I will do next

Phase 3 pivots from “clean the past” to **predict the future**.

I will:

1. Split the cleaned history **in time order** (earlier for training, later for honest testing)—no shuffling that would leak the future into the past  
2. Establish simple forecasting baselines (**ARIMA** and **Prophet**) so we have an explainable floor  
3. Move to stronger models (**XGBoost**, then **LSTM**) under the same evaluation rules  
4. Report forecast quality in business-readable error terms (how far off typical predictions are on the held-out window)  

Same constraints as Phase 2: public data only, reproducible steps, and results I can explain without hiding behind jargon.

---

## 9. Closing

In Phase 2 I turned a trusted smart-meter timeline into a **context-aware detection and cleaning system**: production Isolation Forest as the default watchdog, a documented clean history of **5,000** continuous half-hours, and a full research track that proves we can improve detection on unseen future periods—while keeping the default file stable until leadership chooses otherwise.

We are ready for Phase 3 forecasting on that cleaned history. I recommend proceeding with the default clean artifact now, and holding the enhanced cleaning recipe for a later, explicit decision if you want to optimize how aggressively we rewrite suspicious intervals.

I am available to walk through any section of this report or the supporting project documentation in more detail.

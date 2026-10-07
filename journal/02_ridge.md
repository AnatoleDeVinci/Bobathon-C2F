# 02_ridge

<!--
Design note for experiments/02_ridge.py (same stem, one to one
with the script).
-->

## Question / hypothesis

Does a simple linear model (`Ridge`) on numeric clinical features beat the dummy mean,
and by how much?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Run a linear model and push the report to the hub. Please compare the results to the dummy model."
  - Lab guide `docs/DAY2.md` § 1 "A simple linear model".
- **Why this matters:** The dummy sets the error floor at RMSE 16.57. Ridge tests whether the numeric features carry any additive linear signal at all. EDA showed `off` (r=0.87) and `on` (r=0.67) are strongly correlated with target — a linear model should capture a large portion of that.

## Method

- **Files touched:** `experiments/02_ridge.py`, `submissions/02_ridge.csv`
- **Change versus baseline (or previous experiment):** Replace `DummyRegressor` with
  `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))`. Median-impute
  first because Ridge cannot handle NaN (≈37–79% missingness in several columns).
  String columns (`patient_id`, `cohort`, `gene`, `rater_id`) are excluded from
  features — Ridge needs numbers only.
  Evaluate as a **comparison report**: `{"dummy": dummy, "ridge": ridge}` so both
  appear side-by-side on the same Skore Hub page.
- **Cross-validation:** `splitter=0.2` — same random 80/20 row holdout as `01_dummy`
  (lab guide mandates this for `02_ridge`; patient-grouped CV is deferred to step 3).
- **Out of scope for this experiment:** `alpha` tuning, categorical features (`gene`,
  `cohort`), patient-grouped CV, HistGradientBoosting.

## Risks / things that could invalidate the result

- The same random row holdout leaks patients across train/test (same patient's visits
  on both sides). This inflates Ridge's apparent score relative to the Kaggle grouped
  holdout. The comparison vs dummy is still valid for "does the model learn something?"
  but the absolute RMSE number is optimistic.
- `off` and `on` may be target-derived (leakage risk flagged in EDA). If they are
  available at test time, the Ridge result reflects that; if not, the production score
  would be lower.
- Median imputation treats missingness as noise. If missingness is informative (EDA
  suggests it is — `time_since_intake_off` 79% missing, often for ON-only visits),
  imputing may hide signal. HistGBRT with native NaN handling is the next step.

## Status

- **State:** done
- **Approved by user on:** 2025-07-10
- **Headline result:** Ridge RMSE 10.67, MAE 8.57 (80/20 row holdout) — 35.6% improvement over dummy (RMSE 16.57). Hub report: https://skore.probabl.ai/c2f/bobathon-esilv/estimators/46756
- **Implication for next iteration:** The numeric features carry strong linear signal (`off` r=0.87 drives most of it). Next: patient-grouped CV to get a more honest estimate of generalization, and/or HistGradientBoosting to exploit missingness as signal rather than imputing it away.

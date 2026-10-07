# 06_patient_features

<!--
Design note for experiments/06_patient_features.py (same stem, one to one
with the script).
-->

## Question / hypothesis

Do stateless patient-level aggregation features — disease duration, visit
ordering, per-patient statistics of `off`/`on`/`ledd`, deviation from patient
mean, the ON−OFF gap, and lag/lead visit scores — improve over
`tabular_pipeline` on raw columns (RMSE 7.51 ± 0.12)?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request specifying the exact feature set to compute.
  - CONTEXT.md analysis: temporal progression and drug timing are the two named
    pitfalls that require patient-level context to address.
- **Why this matters:** Every model so far treats visits as independent rows. The
  context document identifies temporal progression (per-patient trajectory) and
  pharmacokinetics (within-visit drug timing) as the two main pitfalls. Patient-level
  aggregations give the model a patient's own history as context without any
  cross-fold leakage, because all features are computed separately on train and
  test frames (no target used; no information flows from test labels into train).

## Method

- **Files touched:** `src/parkinson/features.py` (new), `experiments/06_patient_features.py`,
  `submissions/06_patient_features.csv`
- **Change versus previous experiment:** Add a `add_patient_features(df)` function
  (pure pandas, stateless, no target column) that computes the following on the
  input frame grouped by `patient_id`, with visits sorted by `age`:
  - `disease_duration` = `age − age_at_diagnosis`
  - `visit_number` = rank of this visit within patient (1-based, sorted by `age`)
  - `n_visits` = total number of visits for this patient in the frame
  - `off_measured_frac` = fraction of visits with non-null `off`
  - Per-patient mean/median/std/min/max of `off`, `on`, `ledd`
    → columns `off_mean`, `off_median`, `off_std`, `off_min`, `off_max` (etc.)
  - `off_dev` = `off − off_mean` (deviation from patient mean); same for `on`, `ledd`
  - `on_minus_off` = `on − off` (ON/OFF gap at this visit)
  - `prev_off`, `prev_on` = previous visit's `off`/`on` (NaN for first visit)
  - `next_off`, `next_on` = next visit's `off`/`on` (NaN for last visit)
  - `age_gap_prev` = `age − age_prev` (NaN for first visit)
  - `age_gap_next` = `age_next − age` (NaN for last visit)
  
  Function is applied separately to `X_train_raw` and `X_test` before joining
  with `y_train`. `patient_id` and `Index` are dropped before the model sees
  the features. `tabular_pipeline("regressor")` handles all types and missingness.
  Same `GroupKFold(5)` cv_splits as experiments 03–05.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id`.
- **Out of scope:** lag features that require test-fold history from train (any
  such look-up would cross the fold boundary); target-derived features.

## Risks / things that could invalidate the result

- `prev_off`/`next_off` use only within-frame ordering by `age`; for test patients
  all visits are unknown at prediction time (no train history), so these lags are
  derived from the test frame itself — that is valid (the test CSV contains all
  visits for a patient; we sort and lag within those rows only).
- Per-patient stats (`off_mean` etc.) for test patients are computed from the test
  frame only, which is smaller (~2 visits/patient on average vs ~8 in train). The
  stats will be noisier for test, but this is unavoidable and not leakage.
- `n_visits` will be ~8 for train patients and ~2 for test patients on average —
  the model may learn this difference as a spurious signal.

## Status

- **State:** approved
- **Approved by user on:** 2025-07-10
- **Headline result:** n/a — not yet run
- **Implication for next iteration:** n/a — not yet run

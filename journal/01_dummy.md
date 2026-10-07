# 01_dummy

<!--
Design note for experiments/01_dummy.py (same stem, one to one
with the script).
-->

## Question / hypothesis

What is the error floor for this problem — how wrong are we when the model
knows nothing about individual patients and simply predicts the training-set
mean for every visit?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Run a dummy regressor from sklearn to act as my baseline model. Push to the hub with the name '01_dummy' as the report name to the project 'bobathon-esilv'."
- **Why this matters:** Any real model must beat this floor. If a smart model scores the
  same as or worse than the dummy, the data load, metric, or submission is broken —
  not the model. A dummy baseline also gives RMSE a reference scale: "RMSE 8 compared
  with 12 for the dummy" means the model removes a third of the unexplained variance.

## Method

- **Files touched:** `src/parkinson/__init__.py`, `src/parkinson/hub.py`,
  `experiments/01_dummy.py`, `submissions/01_dummy.csv`
- **Change versus baseline (or previous experiment):** First experiment; no prior to
  compare against. `DummyRegressor(strategy="mean")` ignores all features and always
  predicts the training-set average. Evaluated with `skore.evaluate` using
  `splitter=0.2` (random 80/20 row holdout — not patient-grouped, as per the lab guide;
  the patient leakage is intentional at this stage to establish a row-level floor).
- **Cross-validation:** `splitter=0.2` — random row holdout, no grouped split.
  Patient-grouped CV is deferred to the next experiment. (Lab guide note: `01_dummy`
  and `02_ridge` use the row holdout explicitly; the grouped CV discussion happens later.)
- **Out of scope for this experiment:** any feature engineering, imputation, or model
  selection; patient-grouped splits.

## Risks / things that could invalidate the result

- The random 80/20 holdout splits patient visits across train and test — some patients
  appear on both sides. This inflates the dummy's apparent performance slightly compared
  with a proper patient-holdout, but the dummy's RMSE is mechanically close to
  `std(target)` regardless, so the effect is minimal.
- The Kaggle test set is patient-holdout (`patient_id` does not overlap train). The
  dummy's Kaggle submission is the training-mean constant, which is a valid submission
  but will score exactly `std(y_test)` if the test distribution matches train.

## Status

- **State:** done
- **Approved by user on:** 2025-07-10
- **Headline result:** RMSE 16.57, MAE 13.62 (random 80/20 row holdout) — as expected, ≈ std(target) = 16.5. Hub report: https://skore.probabl.ai/c2f/bobathon-esilv/estimators/46331
- **Implication for next iteration:** The error floor is RMSE ≈ 16.6. Any real model must beat this. Next step: Ridge regression with the same row holdout to see whether the features carry signal at all (expected large improvement given `off` r=0.87 with target).

# 03_grouped_cv

<!--
Design note for experiments/03_grouped_cv.py (same stem, one to one
with the script).
-->

## Question / hypothesis

How much does patient-grouped CV change the estimated RMSE for dummy and Ridge
compared with the random 80/20 row holdout used in experiments 01 and 02?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Re-evaluate dummy and ridge with patient-grouped cross-validation (GroupKFold, 5 splits, groups=patient_id) and push the comparison to the hub as 03_grouped_cv."
  - Lab guide `docs/DAY2.md` § 3 "Evaluate with patient-grouped CV".
- **Why this matters:** The random 80/20 holdout in experiments 01 and 02 leaks patients across train/test (the same patient's visits appear on both sides), inflating the apparent performance. `GroupKFold` is a dress rehearsal of the Kaggle split: a whole patient goes to train or to the held-out fold, never both. The difference between random-split and grouped-split RMSE reveals how much the earlier metrics were optimistic.

## Method

- **Files touched:** `experiments/03_grouped_cv.py`
- **Change versus previous experiment:** Same models (dummy and Ridge from 02) re-evaluated with `GroupKFold(n_splits=5)` on `patient_id`. Splitter is precomputed as `cv_splits = list(GroupKFold(5).split(X, y, groups=groups))` and passed as `splitter=cv_splits` to `skore.evaluate`. Hub's `project.put` accepts `CrossValidationReport` only (not `ComparisonReport`), so dummy and ridge are evaluated separately and pushed under two keys: `03_grouped_cv_dummy` and `03_grouped_cv_ridge`.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` — this is `G-CV-SPLITTER` resolved; recorded in Workspace decisions.
- **Out of scope for this experiment:** new features, categorical encoding, HistGradientBoosting.

## Risks / things that could invalidate the result

- `GroupKFold` still uses the same feature set as experiment 02 (numeric columns only, median impute). The grouped RMSE reflects the grouped generalization but not the effect of better features.
- 5-fold GroupKFold with 5,576 patients gives ~1,115 held-out patients per fold — large enough for stable estimates.
- `off` and `on` collinearity with target may still reflect leakage (same visit measurement); this experiment does not address that.

## Status

- **State:** done
- **Approved by user on:** 2025-07-10
- **Headline result:** Grouped CV (GroupKFold 5) — dummy RMSE 16.50 ± 0.31, Ridge RMSE 10.60 ± 0.24; Ridge −35.7% vs dummy. Hub: dummy https://skore.probabl.ai/c2f/bobathon-esilv/cross-validations/46972 · ridge https://skore.probabl.ai/c2f/bobathon-esilv/cross-validations/46984
- **Implication for next iteration:** Patient-grouped CV barely changes the RMSE vs the random row holdout (16.50 vs 16.57 for dummy; 10.60 vs 10.67 for ridge) — the earlier row-holdout estimates were already honest. Ridge still wins by ~36%. Next: add string features (gene, cohort) and/or switch to HistGradientBoosting to handle missingness natively.

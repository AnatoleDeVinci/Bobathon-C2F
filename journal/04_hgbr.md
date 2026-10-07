# 04_hgbr

<!--
Design note for experiments/04_hgbr.py (same stem, one to one
with the script).
-->

## Question / hypothesis

Does `HistGradientBoostingRegressor` — which handles NaN natively and learns
whether a missing value predicts high or low `target` — beat Ridge under
patient-grouped CV?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Train a HistGradientBoostingRegressor keeping missing values as
    NaN (no imputation), evaluate it with the same patient-grouped GroupKFold splits
    as before, compare it with ridge, and push to the hub as 04_hgbr."
  - Lab guide `docs/DAY2.md` § 4 "HistGradientBoosting: missingness as signal".
- **Why this matters:** Ridge had to median-impute ≥37–79% missing values in several
  columns, discarding any information carried by the absence of a measurement.
  HistGBR routes NaN values down a learned branch at every split — "OFF score missing"
  can be a genuine clinical signal (ON-only visits). The same grouped CV from `03`
  gives an apples-to-apples comparison.

## Method

- **Files touched:** `experiments/04_hgbr.py`, `submissions/04_hgbr.csv`
- **Change versus previous experiment:** Replace `make_pipeline(SimpleImputer, Ridge)`
  with `HistGradientBoostingRegressor(random_state=0)` applied directly to `X_train`
  (no imputation step). Same `FEATURE_COLS` as experiments 01–03 (numeric columns only;
  `gene` / `cohort` / `rater_id` deferred). Same `cv_splits` from
  `GroupKFold(5).split(X, y, groups=patient_id)`. Ridge is re-evaluated on the same
  splits for a direct comparison, and both reports are pushed separately (`04_hgbr`
  for HGBR, `04_hgbr_ridge` for Ridge re-evaluated on this run's splits).
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` — same as `03`.
- **Out of scope for this experiment:** categorical features (`gene`, `cohort`),
  hyperparameter tuning, `tabular_pipeline`.

## Risks / things that could invalidate the result

- Using only numeric columns means `gene` / `cohort` are still excluded; the HGBR
  vs Ridge comparison reflects numeric-only performance, not the full feature space.
- `off` and `on` may still be target-derived (leakage risk from EDA); results would
  look better than production if those features are not available at prediction time.
- HistGBR has more hyperparameters than Ridge; `random_state=0` makes runs
  reproducible but `max_iter`, `learning_rate`, etc. are at defaults.

## Status

- **State:** done
- **Approved by user on:** 2025-07-10
- **Headline result:** GroupKFold(5) — HGBR RMSE 7.76 ± 0.14, Ridge RMSE 10.60 ± 0.24; HGBR −26.8% vs Ridge, −52.9% vs dummy floor (16.50). Hub: HGBR https://skore.probabl.ai/c2f/bobathon-esilv/cross-validations/47203 · Ridge https://skore.probabl.ai/c2f/bobathon-esilv/cross-validations/47209
- **Implication for next iteration:** Native NaN handling (missingness as signal) gives HGBR a large win over Ridge. Next: add categorical features (`gene`, `cohort`) via `skrub.tabular_pipeline` or `TableVectorizer` to let the model exploit all columns.

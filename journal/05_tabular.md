# 05_tabular

<!--
Design note for experiments/05_tabular.py (same stem, one to one
with the script).
-->

## Question / hypothesis

Does including the string categorical columns (`gene`, `cohort`, `rater_id`)
via `skrub.tabular_pipeline("regressor")` improve over the numeric-only HGBR (RMSE 7.76)?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Use skrub tabular_pipeline('regressor') on all columns except
    Index, patient_id and target, evaluate with the same GroupKFold splits and push
    as 05_tabular."
  - Lab guide `docs/DAY2.md` § 5 "skrub `tabular_pipeline`: mixed types without
    hand-encoding".
- **Why this matters:** Experiments 01–04 used only the 8 numeric columns, leaving
  `gene` (4 categories, 32% missing), `cohort` (2 categories), and `rater_id`
  (50 raters) on the table. `tabular_pipeline` picks an encoder per column
  automatically — one-hot for low-cardinality, `StringEncoder` for high-cardinality —
  so we can add all non-identifier columns without any hand-crafting. If those
  string columns carry signal the groupby structure of `rater_id` or the genetic
  mutation recorded in `gene` could reduce RMSE further.

## Method

- **Files touched:** `experiments/05_tabular.py`, `submissions/05_tabular.csv`
- **Change versus previous experiment:** Replace `HistGradientBoostingRegressor`
  (numeric-only) with `skrub.tabular_pipeline("regressor")` applied to all columns
  except `Index`, `patient_id`, and `target`. Same `cv_splits` from
  `GroupKFold(5).split(X_full, y, groups=patient_id)`. Report pushed as `05_tabular`.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` — same as 03 and 04.
- **Out of scope for this experiment:** hyperparameter tuning, skrub DataOps graph,
  custom encoders per column.

## Risks / things that could invalidate the result

- `rater_id` has 50 categories and no missing values; if rater identity strongly
  predicts `target` (rater effect rather than patient effect), it would inflate CV
  scores. The feature importance on the hub will reveal this.
- `gene` is 32% missing; its missingness may already be captured implicitly by HGBR's
  NaN routing. Including it as a categorical adds explicit label signal but the
  interaction with missingness may be complex.
- `tabular_pipeline("regressor")` defaults to `HistGradientBoostingRegressor`
  internally — so this experiment tests whether adding the categorical columns on top
  of HGBR's default estimator helps, not a different estimator family.

## Status

- **State:** done
- **Approved by user on:** 2025-07-10
- **Headline result:** GroupKFold(5) — tabular_pipeline RMSE 7.51 ± 0.12, MAE 5.92; +3.1% vs numeric-only HGBR (7.76 ± 0.14). Hub: https://skore.probabl.ai/c2f/bobathon-esilv/cross-validations/47332
- **Implication for next iteration:** Adding `gene`, `cohort`, and `rater_id` gives a modest but consistent improvement. The categorical columns carry a small additional signal beyond the numeric features HGBR already exploits via NaN routing. Further gains likely require feature engineering (disease duration, visit-level time ordering) or hyperparameter tuning.

<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed - do not invent facts. Keep "Modelling implications" as
candidate suggestions to weigh when designing the model, not final
decisions.
-->

# EDA: Parkinson's Disease UPDRS Motor Score Prediction

_Generated from `data/eda.py` on 2025-07-10._

## Dataset at a glance

- **Tables:** 1 training table (X_train + y_train joined), 1 test table (X_test)
- **Shape:** 44,590 × 13 (train, including target) · 11,013 × 12 (test, no target)
- **Target:** `target` — continuous UPDRS-derived motor score (regression)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | dtype | Missing % | n_unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5,576 | High-cardinality entity ID; 44,590 rows from 5,576 patients → multiple rows per patient |
| `cohort` | string | 0 % | 2 | Binary cohort indicator (A / B) |
| `sexM` | int | 0 % | 2 | Binary sex encoding (0/1) |
| `gene` | string | 32.4 % | 4 | **Most missing feature** — almost a third of rows have no gene label; 4 categories |
| `age_at_diagnosis` | float | 5.2 % | 563 | Small missingness |
| `age` | float | 0 % | 1,150 | Current age; see Associations for collinearity with `age_at_diagnosis` |
| `ledd` | float | 36.6 % | 1,320 | Levodopa equivalent daily dose — **significant missingness** (37 %) |
| `time_since_intake_on` | float | 46.4 % | 64 | Time since medication intake (ON state) — **nearly half missing** |
| `time_since_intake_off` | float | 78.8 % | 178 | Time since medication intake (OFF state) — **most missing** (79 %) |
| `rater_id` | string | 0 % | 50 | 50 distinct raters; potential rater effect |
| `on` | float | 29.6 % | 83 | UPDRS-III score measured in ON state — 30 % missing |
| `off` | float | 42.4 % | 101 | UPDRS-III score measured in OFF state — 42 % missing |
| `target` | float | 0 % | 867 | Prediction target — no missing values |

**Notable findings:**
- Four features have substantial missingness: `time_since_intake_off` (79 %), `time_since_intake_on` (46 %), `ledd` (37 %), `gene` (32 %). Any model must handle these; naïve median imputation may not be appropriate given their clinical meaning.
- `patient_id` repeats: 44,590 rows over 5,576 patients = ~8 rows per patient on average. This is the key structural signal for cross-validation (see Modelling implications).
- `rater_id` introduces 50 different raters — a potential source of systematic bias.
- No constant or clearly erroneous columns.

## Target

**Continuous regression target.**

| Statistic | Value |
|---|---|
| Mean | 37.47 |
| Std | 16.50 |
| Min | 0.0 |
| Q25 | 25.6 |
| Median | 37.3 |
| Q75 | 49.3 |
| Max | 109.5 |

The distribution is roughly bell-shaped (centred near 37) with a right tail extending to 109.5. The IQR is 23.7. No heavy skew, but the right tail with very high scores (>80) is sparse. No missing values in the target.

## Structure

**No datetime columns** were detected by skrub (no date-typed columns in the dataset).

**Group / id columns detected:**

| Column | Unique ratio |
|---|---|
| `patient_id` | 12.5 % (5,576 unique values / 44,590 rows) |
| `ledd` | 3.0 % |
| `age` | 2.6 % |
| `rater_id` | 0.11 % |

`patient_id` is the dominant group signal: multiple rows per patient means a standard KFold will leak patient data across folds (a patient's visits will appear in both train and validation). `rater_id` is a secondary grouping concern.

## Associations

**Strongest links with `target`:**

| Feature | Pearson r | Cramér V | Notes |
|---|---|---|---|
| `off` | **0.871** | 0.406 | Very strong linear link — UPDRS OFF-state score is the dominant predictor |
| `on` | **0.669** | 0.368 | Strong — UPDRS ON-state score, second predictor |
| `age` | 0.310 | 0.114 | Moderate |
| `ledd` | 0.298 | 0.236 | Moderate |
| `age_at_diagnosis` | 0.133 | 0.051 | Weak |
| `time_since_intake_off` | 0.008 | 0.092 | Near-zero linear; some non-linear signal (Cramér V) |
| `time_since_intake_on` | 0.000 | 0.177 | No linear signal, some non-linear |
| `cohort` | nan | 0.090 | Categorical; modest signal |

**⚠ Leakage risk — `off` and `on`:** Both `off` (Pearson r = 0.87) and `on` (r = 0.67) are UPDRS-III subscores. The `target` is very likely derived from or highly dependent on these scores clinically. If `off` and `on` are available at prediction time, they are legitimate features; if they are **not** available in the real use case (e.g., the model is supposed to predict the score *before* those measurements are taken), they would be leakage. **This needs clarification before modelling.**

**Strong feature–feature collinearity:**

| Pair | Pearson r | Notes |
|---|---|---|
| `age_at_diagnosis` ↔ `age` | **0.942** | Near-perfect collinearity — one may be redundant |
| `on` ↔ `off` | **0.872** | Strongly correlated subscores |

## Modelling implications

- **`patient_id` repeats across rows → use `GroupKFold` with `groups=patient_id`** to ensure patient visits are not split across train and validation. A standard KFold would give over-optimistic CV scores (patient-level leakage).
- **No datetime column → `TimeSeriesSplit` is not indicated** unless longitudinal ordering within a patient is relevant (open question below).
- **Missingness is high on several features** (`time_since_intake_off`, `ledd`, `gene`) → skrub's `tabular_pipeline` default encoders handle missing values natively; a custom imputation strategy may be needed for `off` and `on` (which are UPDRS scores, not random-missing).
- **`off` and `on` are very strong predictors (r > 0.67)** → if they are legitimate features, the model will largely be "predicting target from subscores." Clarify whether these are available at prediction time.
- **`age_at_diagnosis` ↔ `age` collinearity (r = 0.94)** → consider dropping one or letting the model handle it; no action required for tree-based models but may matter for linear models.
- **Right-tailed target distribution** → R² and MAE are both reasonable metrics; no urgent need for a target transform, but worth monitoring residuals at high score values.
- **`rater_id` has 50 categories and no missingness** → a good candidate for a `high_cardinality_encoder` in skrub's pipeline; also a potential nuisance variable to encode or control for.

## Open questions

1. **`off` and `on` leakage:** Are these UPDRS subscores available at prediction time, or does the model need to predict `target` *without* them? The answer changes the feature set and the expected performance ceiling dramatically.
2. **Target definition:** Is `target` the total MDS-UPDRS-III score, an average of ON and OFF, or something else? Understanding its exact derivation clarifies whether `on`/`off` are legitimate features.
3. **Longitudinal ordering within a patient:** Are the ~8 rows per patient longitudinal visits in time order? If so, `TimeSeriesSplit` within a patient group or a `GroupShuffleSplit` may be more appropriate than plain `GroupKFold`.
4. **`rater_id` confound:** Should rater be treated as a nuisance variable (encode + include), dropped, or explicitly modelled (mixed-effects style)?
5. **`gene` missingness (32 %):** Is missingness in `gene` informative (i.e., patients without a known mutation are a distinct subgroup), or is it genuinely unknown?

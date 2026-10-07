# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease UPDRS motor score regression
- **Goal:** Predict a continuous UPDRS-derived motor score from patient clinical features
- **Last experiment:** 05_tabular — done
- **Last result:** tabular_pipeline RMSE 7.51 ± 0.12 (GroupKFold-5) — +3.1% vs HGBR, −54.5% vs dummy

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-10
  - env manager: <pixi | uv | poetry | hatch | conda | pip+venv> - recorded: <YYYY-MM-DD>
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: <name1, name2 | none> - recorded: <YYYY-MM-DD>
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: <local | hub | mlflow> - recorded: <YYYY-MM-DD>
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: GroupKFold - recorded: 2025-07-10

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done - 2025-07-10
- **Summary:** 44,590 training rows from 5,576 patients (≈8 visits/patient) × 12 features; continuous regression target (mean 37.5, std 16.5). `off` (r=0.87) and `on` (r=0.67) are dominant predictors but may be leakage — needs clarification. `patient_id` repeats across rows → `GroupKFold` is the candidate splitter. High missingness on `time_since_intake_off` (79%), `ledd` (37%), `gene` (32%).
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor mean — error floor | done | RMSE 16.57, MAE 13.62 (80/20 row holdout) | [01_dummy.md](01_dummy.md) |
| `02_ridge` | Ridge + median imputation vs dummy | done | RMSE 10.67, MAE 8.57 (80/20 row holdout) — −35.6% vs dummy | [02_ridge.md](02_ridge.md) |
| `03_grouped_cv` | Dummy & Ridge re-evaluated with GroupKFold(5) on patient_id | done | Ridge RMSE 10.60 ± 0.24, dummy 16.50 ± 0.31 (grouped CV) | [03_grouped_cv.md](03_grouped_cv.md) |
| `04_hgbr` | HistGBR (native NaN) vs Ridge under GroupKFold(5) | done | HGBR RMSE 7.76 ± 0.14, Ridge 10.60 ± 0.24 — HGBR −26.8% vs Ridge | [04_hgbr.md](04_hgbr.md) |
| `05_tabular` | skrub tabular_pipeline (all cols, mixed types) vs HGBR | done | RMSE 7.51 ± 0.12 — +3.1% vs numeric HGBR | [05_tabular.md](05_tabular.md) |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note.
-->

| # | Item | Source |
|---|---|---|
| <!-- B1 --> | <!-- placeholder --> | <!-- placeholder --> |

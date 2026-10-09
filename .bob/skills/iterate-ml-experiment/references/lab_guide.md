# Lab guide

Load this when `docs/GUIDED.md` exists in the project. The guide and
`docs/CONTEXT.md` are the brief. This file does not restate their
steps. Read both documents, then apply the rules below.

If `docs/GUIDED.md` is absent, ignore this file and keep the generic
experiment loop.

## Goal

Propose this sentence, and let the student amend it:

> minimize RMSE for the true OFF MDS-UPDRS motor score, on patients
> held out from training

Clinic ON/OFF scores are biased. The target column is the unbiased
score. The Kaggle holdout is by `patient_id`, not by visit.

## EDA

Each student runs EDA in their own workspace. Follow the regular
`G-EDA` flow in `explore-ml-data`: **run** / **skip**, or **re-run** /
**keep** when `data/eda.md` already exists. On **run**, that skill
writes `data/eda.py`, `data/eda.md`, and `data/eda_<table>.html`
locally. Hub keys are model reports (`01_dummy`, `02_ridge`, …).

## What to propose next

Infer the unfinished section from artifacts. Suggest that section.
Do not block a student who names a later model; say which guide
section that request skips, then do what they asked.

| Evidence the section is still open | Suggest |
|---|---|
| No `data/eda.md` | Local EDA (`G-EDA`) |
| EDA present, no hub report `01_dummy` | Dummy mean, RMSE, row holdout |
| `01_dummy` present, no `02_ridge` | Ridge on the same row holdout |
| `02_ridge` present, no `submissions/02_ridge.csv` | The guide's Kaggle upload for that Ridge |
| A row-holdout Ridge report exists, and no grouped-CV report does | Patient-grouped CV |
| Those are done | The next section heading in `docs/GUIDED.md` |

Hub report keys come from the guide (`01_dummy`, `02_ridge`, …).
One new key per Kaggle file.

## Row holdout, then grouped CV

`01_dummy` and `02_ridge` call `skore.evaluate` with `splitter=0.2`
(a random visit holdout). Do not ask `G-CV-SPLITTER` for those two.
The leak (the same `patient_id` on both sides) is the point of that
holdout.

`G-CV-SPLITTER` opens on the guide's patient-grouped section, or
when the student asks for a grouped split earlier. The guide's
family is `GroupKFold` on `patient_id`. Record the student's answer
if they pick a different family.

## Kaggle file

Each experiment's submission is written to
`submissions/<experiment-key>.csv` - the script's stem
(`experiments/02_ridge.py` → `submissions/02_ridge.csv`), which is
also the estimator's name. One CSV per experiment: filename = stem =
hub report key, so a later model never overwrites an earlier
submission. Header `Index,target`, one row per test visit. The fit
uses **all** training visits (no CV split) - not the CV fit behind
the report.

The Kaggle submission description must contain the
`https://ibm.skore.probabl.ai/…` URL printed for that model's hub
report. A score without that URL does not count.

# Explore ML Data: Cell anatomy

Concrete cell shapes for `data/eda.py`, the `TableReport` repr trap,
the library-agnostic (skrub) approach, and how each finding maps to a
downstream modelling gate. SKILL.md carries the compact cell
sequence; load this when you are actually writing or debugging the
cells.

## Two locations, kept separate

| Concern | Variable | Where | Mutable? |
|---|---|---|---|
| Raw data source | `RAW = <LOAD_RAW_DATA>` | anywhere - `data/`, `raw/`, an absolute path, or external | **read-only** |
| EDA deliverables | `EDA_DIR` | always `PROJECT_ROOT / "data"` (created if missing) | written by this skill only |

The raw data is **never** assumed to live in `data/`. Only the
deliverables (`eda_<table>.html`, and `eda.md` authored by the agent)
land in `EDA_DIR`. This is the fix for "data lives in another folder".

## Library-agnostic by design: read facts off skrub

The workspace's tabular library (G-TABULAR) may be pandas **or**
polars, whose summary methods differ (`isna`/`null_count`,
`nunique`/`n_unique`, `select_dtypes` doesn't exist in polars, …).
Writing pandas-specific code here breaks on polars workspaces.

`skrub` is the equaliser: `TableReport` and `column_associations`
accept both libraries and return the same thing. So the structured
facts (dtypes, missingness, cardinality, datetime inference, target
summary) come from `report.json()`, **not** from dataframe methods.
The only library-specific line in the whole file is
`RAW = <LOAD_RAW_DATA>`.

> The template's `TableReport.json()` keys are `name`, `dtype`,
> `null_proportion`, `n_unique`, `n_rows`, and `columns`. Run
> `data/eda.py` first. Write a scratch probe only if the digest shows
> those fields as null. Do not probe before the first run.

## The `TableReport` repr trap (the load-bearing rule)

`skrub.TableReport` is built to render rich HTML in a notebook. The
shared runner is **not** a notebook - it captures each cell's last
bare expression via `repr(result.result)`. A bare `TableReport`
reprs to nothing useful:

```python
# WRONG: digest shows: <TableReport: use .open() to display>
skrub.TableReport(RAW)
```

```python
# RIGHT: write the rich HTML (statement), read facts from json()
report = skrub.TableReport(RAW, title="customers", verbose=0)
report.write_html(EDA_DIR / "eda_customers.html")
summary = json.loads(report.json())
{"n_rows": summary.get("n_rows"), "n_columns": len(summary.get("columns", []))}
```

`verbose=0` is load-bearing too: the default `verbose=1` prints
per-column progress into the cell's `stdout:` section.

## Bare expressions, not `print()`

```python
# WRONG: lands in stdout, mixed with other noise, harder to scan
print(summary["n_rows"])
```

```python
# RIGHT: captured in the cell's **output:** section
{"n_rows": summary.get("n_rows")}
```

Statement-only cells (assignments, `write_html(...)`,
`EDA_DIR.mkdir(...)`) are fine - they just produce no `output:`
section. Put the value you want to read on the **last** line.

## Cell-by-cell, with downstream mapping

### Cell 2: imports + paths

```python
import json
from pathlib import Path

import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)
```

The runner sets `__file__` to `data/eda.py`, so this locates the repo
before `src/<pkg>/` exists. A missing package import is not a reason
to scaffold. No pandas/polars import here - only the load cell needs
the tabular lib.

### Cell 3: load raw data (anywhere)

```python
RAW = pd.read_parquet(PROJECT_ROOT / "data" / "train.parquet")  # or elsewhere
RAW.shape
```

Adapt the load to where the data actually lives - in-repo folder,
absolute path, or external store. For multiple tables, load each into
its own variable and repeat the overview cell (Cell 4) per table, one
HTML file each.

### Cell 4: overview → downstream: learner / encoders

```python
report = skrub.TableReport(RAW, title="train", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": c.get("name"),
        "dtype": c.get("dtype"),
        "null_pct": c.get("null_proportion"),
        "n_unique": c.get("n_unique"),
    }
    for c in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}
```

- **High `null_pct`** → note columns that may need imputation /
  dropping in the pipeline (not here).
- **High `n_unique` on string columns** → high-cardinality
  categoricals; skrub's default encoders handle these. Free-text
  columns may want a text encoder - flag it.

### Cell 5: target → downstream: metric + stratification

```python
TARGET = "<TARGET_COLUMN>"
next((c for c in summary.get("columns", []) if c.get("name") == TARGET), None)
```

The target's column entry carries value counts (low-cardinality →
classification) or a distribution summary (numeric → regression), so
one expression covers both task types - no `value_counts` blow-up on a
continuous target.

- **Imbalance** → implication: `StratifiedKFold` + ROC-AUC / PR-AUC
  over accuracy.
- **Heavy skew** → implication: candidate target transform; flag in
  the baseline note's Risks.

### Cell 6: structure → downstream: `G-CV-SPLITTER`

```python
datetime_cols = [
    c.get("name") for c in summary.get("columns", [])
    if "date" in str(c.get("dtype", "")).lower()
]
unique_ratio = sorted(
    ({"column": c.get("name"),
      "unique_ratio": (c.get("n_unique") or 0) / n_rows if n_rows else None}
     for c in summary.get("columns", [])),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}
```

- **Datetime column present + forecasting task** → implication:
  `TimeSeriesSplit`. (skrub infers datetimes even from string columns,
  so this catches dates a raw `select_dtypes` would miss.)
- **A column whose values repeat across rows but identify an entity**
  (`user_id`, `patient_id`) → implication: `GroupKFold` on it.
  `unique_ratio` near 1 means a near-unique key (often a row id to
  drop, not a group).

### Cell 7: associations → downstream: leakage check

```python
assoc = skrub.column_associations(RAW)
# Same library as RAW: pandas has to_dict, polars has to_dicts.
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}
```

- A feature with an **implausibly perfect** association to the target
  is a leakage flag - name it in `data/eda.md` § Associations and
  raise it as an open question, do not silently keep it.

## Multiple tables

Run Cell 4 once per table (`eda_<table>.html` each). Run Cells 5–7 on
the **target-bearing** table (the one you will model on); for the
other tables, the Cell 4 overview is usually enough, plus a note on
the join key. Don't try to associate columns across unjoined tables.

## Large data

`TableReport` computes stats over the whole frame and
`column_associations` is roughly O(columns²). On very large datasets,
load a row sample for the report (e.g. the first N rows or a random
sample via the tabular lib) and say so in `data/eda.md`: the goal is
a fast, representative read, not exhaustive stats.

## What NOT to do in these cells

- No imputation / dropping / re-saving of raw files (read-only).
- No `skore.evaluate` / `project.put` (that is the experiment's job).
- No splitter / metric / learner *decision* - only the *evidence*.
- No pandas/polars-specific summary methods - read skrub's json.
- No `warnings.filterwarnings(...)`: stderr in the digest is signal
  (see `python-code-style` § Stop conditions).

## From digest to deliverables

After the run:

1. Read the digest (stdout, or `scratch/eda/eda.md`).
2. Author `data/eda.md` from `templates/eda.md`: every claim
   grounded in the digest; the **Modelling implications** section is
   the payoff the baseline note cites.
3. Write the `journal/JOURNAL.md` § "Data understanding (EDA)" 2–4
   line summary + link to `data/eda.md`.

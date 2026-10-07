"""Lookup: skrub.tabular_pipeline @ installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "skrub"
VERSION = importlib.import_module(LIB).__version__

import skrub

sym = skrub.tabular_pipeline
try:
    sig = str(inspect.signature(sym))
except (TypeError, ValueError):
    sig = "<no signature>"

out = io.StringIO()
out.write(f"# tabular_pipeline\n\n")
out.write(f"Source: inspect: skrub.tabular_pipeline @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")
out.write(f"## Signature\n\n```\n{sig}\n```\n\n")
help_text = pydoc.render_doc(sym, renderer=pydoc.plaintext)
out.write(f"## help()\n\n```\n{help_text[:2000]}\n```\n\n")
out.write(
    "## Usage\n\n"
    "- **Call:** `skrub.tabular_pipeline('regressor')` returns a sklearn-compatible Pipeline\n"
    "- **Don't call:** `skrub.tabular_learner` — renamed to `tabular_pipeline` in 0.7+\n"
    "- **Trap:** pass a DataFrame (not a numpy array) — TableVectorizer needs column names\n"
    "- **Returns:** sklearn Pipeline with TableVectorizer + HistGradientBoostingRegressor\n"
)

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "tabular_pipeline.md").write_text(out.getvalue())
print(out.getvalue())

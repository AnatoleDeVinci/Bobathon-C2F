"""Lookup: Ridge, SimpleImputer, make_pipeline @ sklearn installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "sklearn"
VERSION = importlib.import_module(LIB).__version__

symbols = [
    ("linear_model.Ridge", "Ridge"),
    ("impute.SimpleImputer", "SimpleImputer"),
    ("pipeline.make_pipeline", "make_pipeline"),
]

out = io.StringIO()
out.write(f"# sklearn linear pipeline symbols\n\n")
out.write(f"Source: inspect: sklearn @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")

for dotted, label in symbols:
    parts = dotted.split(".")
    mod = importlib.import_module(f"sklearn.{parts[0]}")
    sym = getattr(mod, parts[1])
    try:
        sig = str(inspect.signature(sym))
    except (TypeError, ValueError):
        sig = "<no signature>"
    out.write(f"## {label}\n\n### Signature\n\n```\n{sig}\n```\n\n")
    help_text = pydoc.render_doc(sym, renderer=pydoc.plaintext)
    out.write(f"### help() (truncated to 800 chars)\n\n```\n{help_text[:800]}\n```\n\n")

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "ridge_pipeline.md").write_text(out.getvalue())
print(out.getvalue()[:2500])
print(f"\nCached to scratch/api/{LIB}/{VERSION}/ridge_pipeline.md")

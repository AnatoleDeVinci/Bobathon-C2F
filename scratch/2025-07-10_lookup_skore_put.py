"""Probe project.put accepted types in skore 0.26.0."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "skore"
VERSION = importlib.import_module(LIB).__version__

# Probe Project.put signature and CrossValidationReport
from skore._project.project import Project
from skore._reports.cross_validation.report import CrossValidationReport

symbols = [
    (Project.put, "Project.put"),
    (CrossValidationReport, "CrossValidationReport"),
]

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)

out = io.StringIO()
out.write(f"# skore project.put and CrossValidationReport\n\n")
out.write(f"Source: inspect: skore @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")

for sym, label in symbols:
    try:
        sig = str(inspect.signature(sym))
    except (TypeError, ValueError):
        sig = "<no signature>"
    out.write(f"## {label}\n\n### Signature\n\n```\n{sig}\n```\n\n")
    help_text = pydoc.render_doc(sym, renderer=pydoc.plaintext)
    out.write(f"### help() (first 600 chars)\n\n```\n{help_text[:600]}\n```\n\n")

(cache_dir / "project_put.md").write_text(out.getvalue())
print(out.getvalue())

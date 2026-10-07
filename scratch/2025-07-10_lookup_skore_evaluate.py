"""Lookup: skore.evaluate and skore.Project @ installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "skore"
VERSION = importlib.import_module(LIB).__version__

symbols = [
    ("evaluate", "evaluate"),
    ("Project", "Project"),
    ("login", "login"),
]

out = io.StringIO()
out.write(f"# skore evaluation and project API\n\n")
out.write(f"Source: inspect: skore @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")

for dotted, label in symbols:
    mod = importlib.import_module(LIB)
    sym = mod
    for part in dotted.split("."):
        sym = getattr(sym, part)
    try:
        sig = str(inspect.signature(sym))
    except (TypeError, ValueError):
        sig = "<no signature>"
    out.write(f"## {label}\n\n### Signature\n\n```\n{sig}\n```\n\n")
    out.write(f"### help()\n\n```\n{pydoc.render_doc(sym, renderer=pydoc.plaintext)}\n```\n\n")

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "evaluate_project.md").write_text(out.getvalue())
print(out.getvalue()[:3000])
print("... (truncated) ...")
print(f"Cached to scratch/api/{LIB}/{VERSION}/evaluate_project.md")

"""Lookup: sklearn DummyRegressor @ installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "sklearn"
VERSION = importlib.import_module(LIB).__version__

mod = importlib.import_module("sklearn.dummy")
sym = mod.DummyRegressor

try:
    sig = str(inspect.signature(sym))
except (TypeError, ValueError):
    sig = "<no signature>"

out = io.StringIO()
out.write(f"# DummyRegressor\n\n")
out.write(f"Source: inspect: sklearn.dummy.DummyRegressor @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")
out.write(f"## Signature\n\n```\n{sig}\n```\n\n")
out.write(f"## help()\n\n```\n{pydoc.render_doc(sym, renderer=pydoc.plaintext)}\n```\n\n")

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "dummy_regressor.md").write_text(out.getvalue())
print(out.getvalue()[:2000])
print(f"\nCached to scratch/api/{LIB}/{VERSION}/dummy_regressor.md")

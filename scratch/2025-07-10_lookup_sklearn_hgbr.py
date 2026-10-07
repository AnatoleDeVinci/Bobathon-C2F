"""Lookup: sklearn.ensemble.HistGradientBoostingRegressor @ installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "sklearn"
VERSION = importlib.import_module(LIB).__version__

from sklearn.ensemble import HistGradientBoostingRegressor

sym = HistGradientBoostingRegressor
try:
    sig = str(inspect.signature(sym))
except (TypeError, ValueError):
    sig = "<no signature>"

out = io.StringIO()
out.write(f"# HistGradientBoostingRegressor\n\n")
out.write(f"Source: inspect: sklearn.ensemble.HistGradientBoostingRegressor @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")
out.write(f"## Signature\n\n```\n{sig}\n```\n\n")
help_text = pydoc.render_doc(sym, renderer=pydoc.plaintext)
out.write(f"## help() (first 1200 chars)\n\n```\n{help_text[:1200]}\n```\n\n")
out.write(
    "## Usage\n\n"
    "- **Call:** `HistGradientBoostingRegressor(random_state=0)` — handles NaN natively, no imputation needed\n"
    "- **Don't call:** with `SimpleImputer` before it — that hides the missingness signal\n"
    "- **Trap:** string columns must be converted to `category` dtype before passing; "
    "pass numeric columns directly\n"
    "- **Returns:** fitted estimator with `.predict(X)` → array of float\n"
)

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "hgbr.md").write_text(out.getvalue())
print(out.getvalue()[:1600])
print(f"\nCached to scratch/api/{LIB}/{VERSION}/hgbr.md")

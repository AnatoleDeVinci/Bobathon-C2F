"""Lookup: sklearn.model_selection.GroupKFold @ installed version."""
from __future__ import annotations

import datetime
import importlib
import inspect
import io
import pydoc
from pathlib import Path

LIB = "sklearn"
VERSION = importlib.import_module(LIB).__version__

from sklearn.model_selection import GroupKFold

sym = GroupKFold
try:
    sig = str(inspect.signature(sym))
except (TypeError, ValueError):
    sig = "<no signature>"

out = io.StringIO()
out.write(f"# GroupKFold\n\n")
out.write(f"Source: inspect: sklearn.model_selection.GroupKFold @ {VERSION}\n")
out.write(f"Probed: {datetime.date.today():%Y-%m-%d}\n\n")
out.write(f"## Signature\n\n```\n{sig}\n```\n\n")
help_text = pydoc.render_doc(sym, renderer=pydoc.plaintext)
out.write(f"## help() (first 1000 chars)\n\n```\n{help_text[:1000]}\n```\n\n")
out.write(
    "## Usage\n\n"
    "- **Call:** `GroupKFold(n_splits=5).split(X, y, groups=patient_id_series)`\n"
    "- **Don't call:** `GroupKFold().split(X, y)` without groups= — raises\n"
    "- **Trap:** skore's sklearn path calls `splitter.split(X, y)` without groups, "
    "so precompute: `cv_splits = list(GroupKFold(5).split(X, y, groups=groups))`\n"
    "- **Returns:** generator of (train_idx, test_idx) arrays\n"
)

cache_dir = Path("scratch/api") / LIB / VERSION
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "group_kfold.md").write_text(out.getvalue())
print(out.getvalue())

"""Data loading and X-marker wiring.

Owns: how raw data is materialized into `(X, y)`, and how structural
metadata (groups, time ordering, ...) is attached at the X marker via
`split_kwargs`. Data paths are decided by the caller — this module
does not invent a `data/` directory.
"""

from __future__ import annotations


def load_dataset():
    """Return `(X, y)` ready for the pipeline.

    Replace the body with the actual loader: `pd.read_parquet`,
    `pd.read_csv`, a fixture fetch, a remote query, etc. The return
    contract is `(X, y)` where `X` is a DataFrame (or whatever the
    pipeline expects at its X marker) and `y` is the target.
    """
    raise NotImplementedError


def load_test_dataset():
    """Return `X_test` for the submission CSV.

    Same columns as the `X` returned by `load_dataset`, minus the
    target; keep the competition's identifier column (`Index`). Apply
    the same cleaning and dtype choices as training so the fitted
    pipeline sees an aligned table.
    """
    raise NotImplementedError

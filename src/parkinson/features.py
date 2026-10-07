"""Patient-level feature engineering for the Parkinson's UPDRS regression task.

All functions are stateless: they compute features from the input frame only,
never using target values and never reading across the train/test boundary.
Apply ``add_patient_features`` separately on the training frame and on the test
frame before fitting or predicting.

Visits are sorted by ``age`` within each patient for all lag/lead and ordering
features.  Missing values in ``off``, ``on``, and ``ledd`` are propagated as
NaN and handled by the downstream ``tabular_pipeline``.
"""

from __future__ import annotations

import pandas as pd


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient aggregation and lag/lead features to ``df``.

    Parameters
    ----------
    df :
        Raw feature frame with at least columns ``patient_id``, ``age``,
        ``age_at_diagnosis``, ``off``, ``on``, ``ledd``.
        Must **not** contain the ``target`` column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional columns appended.  The original
        columns are preserved unchanged.  ``patient_id`` is kept so the
        caller can use it for GroupKFold; drop it before the model if
        required by the pipeline.
    """
    out = df.copy()

    # --- Sort within patient by age so lag/lead are chronological ----------
    out = out.sort_values(["patient_id", "age"]).copy()

    # --- Simple stateless features -----------------------------------------
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    # --- Visit ordering within patient -------------------------------------
    out["visit_number"] = (
        out.groupby("patient_id")["age"]
        .rank(method="first")
        .astype(int)
    )
    out["n_visits"] = out.groupby("patient_id")["age"].transform("count")

    # --- Fraction of visits with off measured (per patient) ----------------
    out["off_observed"] = out["off"].notna().astype(float)
    out["off_measured_frac"] = out.groupby("patient_id")["off_observed"].transform(
        "mean"
    )
    out = out.drop(columns=["off_observed"])

    # --- Per-patient statistics for off, on, ledd --------------------------
    for col in ("off", "on", "ledd"):
        grp = out.groupby("patient_id")[col]
        out[f"{col}_mean"] = grp.transform("mean")
        out[f"{col}_median"] = grp.transform("median")
        out[f"{col}_std"] = grp.transform("std")
        out[f"{col}_min"] = grp.transform("min")
        out[f"{col}_max"] = grp.transform("max")
        # Deviation from patient mean
        out[f"{col}_dev"] = out[col] - out[f"{col}_mean"]

    # --- ON minus OFF at this visit ----------------------------------------
    out["on_minus_off"] = out["on"] - out["off"]

    # --- Lag and lead features (previous / next visit, sorted by age) ------
    for col in ("off", "on"):
        out[f"prev_{col}"] = out.groupby("patient_id")[col].shift(1)
        out[f"next_{col}"] = out.groupby("patient_id")[col].shift(-1)

    # Age gap to previous and next visit
    age_shifted_prev = out.groupby("patient_id")["age"].shift(1)
    age_shifted_next = out.groupby("patient_id")["age"].shift(-1)
    out["age_gap_prev"] = out["age"] - age_shifted_prev
    out["age_gap_next"] = age_shifted_next - out["age"]

    # Restore original index order (sort_values changed row order)
    out = out.sort_index()

    return out

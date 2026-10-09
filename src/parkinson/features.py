"""Patient-level feature engineering for the Parkinson's UPDRS regression task.

All functions are stateless: they compute features from the input frame only,
never using target values and never reading across the train/test boundary.
Apply ``add_patient_features`` separately on the training frame and on the test
frame before fitting or predicting.

Visits are sorted by ``age`` within each patient for all lag/lead, ordering,
trend, interpolation, and rolling features.  Missing values in ``off``, ``on``,
and ``ledd`` are propagated as NaN and handled by the downstream
``tabular_pipeline``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_trend(series: pd.Series, ages: pd.Series) -> tuple[float, float]:
    """Return (slope, intercept) for a simple OLS of *series* on *ages*.

    Falls back to (0.0, mean) when fewer than 2 non-null paired values exist.
    """
    mask = series.notna() & ages.notna()
    x = ages[mask].values.astype(float)
    y = series[mask].values.astype(float)
    if len(x) < 2:
        mean_y = float(y.mean()) if len(y) > 0 else np.nan
        return 0.0, mean_y
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    if denom == 0:
        return 0.0, float(y.mean())
    slope = float((x_c * y).sum() / denom)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def _interp_group(grp: pd.DataFrame, col: str) -> pd.Series:
    """Linearly interpolate *col* vs *age* within a patient group.

    For each row, uses the nearest measured visit strictly before and strictly
    after the current age.  Returns NaN when the current visit itself has a
    measured value *or* when no bracket exists on one side.

    The result is a Series aligned to ``grp.index``.
    """
    ages = grp["age"].values
    vals = grp[col].values
    result = np.full(len(grp), np.nan)
    measured_mask = ~np.isnan(vals)
    measured_ages = ages[measured_mask]
    measured_vals = vals[measured_mask]

    for i, age in enumerate(ages):
        before_mask = measured_ages < age
        after_mask = measured_ages > age
        if not before_mask.any() or not after_mask.any():
            continue
        age_b = measured_ages[before_mask][-1]
        age_a = measured_ages[after_mask][0]
        val_b = measured_vals[before_mask][-1]
        val_a = measured_vals[after_mask][0]
        denom = age_a - age_b
        if denom == 0:
            continue
        t = (age - age_b) / denom
        result[i] = val_b + t * (val_a - val_b)

    return pd.Series(result, index=grp.index)


def _nearest_measured_dist(grp: pd.DataFrame, col: str) -> pd.Series:
    """Age distance from each row to the nearest row with a non-null *col*."""
    ages = grp["age"].values
    vals = grp[col].values
    measured_mask = ~np.isnan(vals)
    measured_ages = ages[measured_mask]
    result = np.full(len(grp), np.nan)
    if measured_ages.size > 0:
        for i, age in enumerate(ages):
            result[i] = float(np.min(np.abs(measured_ages - age)))
    return pd.Series(result, index=grp.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient aggregation and lag/lead features to ``df``.

    Parameters
    ----------
    df :
        Raw feature frame with at least columns ``patient_id``, ``age``,
        ``age_at_diagnosis``, ``off``, ``on``, ``ledd``,
        ``time_since_intake_on``, ``time_since_intake_off``.
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

    # --- Sort within patient by age so all sequential features are chronological
    out = out.sort_values(["patient_id", "age"]).copy()

    # -----------------------------------------------------------------------
    # Block 1 (from experiment 06): basic stateless features
    # -----------------------------------------------------------------------

    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    out["visit_number"] = (
        out.groupby("patient_id")["age"]
        .rank(method="first")
        .astype(int)
    )
    out["n_visits"] = out.groupby("patient_id")["age"].transform("count")

    out["off_observed"] = out["off"].notna().astype(float)
    out["off_measured_frac"] = out.groupby("patient_id")["off_observed"].transform(
        "mean"
    )
    out = out.drop(columns=["off_observed"])

    for col in ("off", "on", "ledd"):
        grp = out.groupby("patient_id")[col]
        out[f"{col}_mean"] = grp.transform("mean")
        out[f"{col}_median"] = grp.transform("median")
        out[f"{col}_std"] = grp.transform("std")
        out[f"{col}_min"] = grp.transform("min")
        out[f"{col}_max"] = grp.transform("max")
        out[f"{col}_dev"] = out[col] - out[f"{col}_mean"]

    out["on_minus_off"] = out["on"] - out["off"]

    for col in ("off", "on"):
        out[f"prev_{col}"] = out.groupby("patient_id")[col].shift(1)
        out[f"next_{col}"] = out.groupby("patient_id")[col].shift(-1)

    age_shifted_prev = out.groupby("patient_id")["age"].shift(1)
    age_shifted_next = out.groupby("patient_id")["age"].shift(-1)
    out["age_gap_prev"] = out["age"] - age_shifted_prev
    out["age_gap_next"] = age_shifted_next - out["age"]

    # -----------------------------------------------------------------------
    # Block 2 (new): per-patient linear trend of off and on vs age
    # -----------------------------------------------------------------------
    # slope and fitted value at the current visit age; falls back to patient
    # mean when fewer than 2 non-null measurements exist.

    off_slopes: dict[str, float] = {}
    on_slopes: dict[str, float] = {}
    off_intercepts: dict[str, float] = {}
    on_intercepts: dict[str, float] = {}

    for pid, grp in out.groupby("patient_id"):
        s_off, i_off = _linear_trend(grp["off"], grp["age"])
        s_on, i_on = _linear_trend(grp["on"], grp["age"])
        off_slopes[pid] = s_off
        off_intercepts[pid] = i_off
        on_slopes[pid] = s_on
        on_intercepts[pid] = i_on

    out["off_trend_slope"] = out["patient_id"].map(off_slopes)
    out["on_trend_slope"] = out["patient_id"].map(on_slopes)
    out["off_trend_fitted"] = (
        out["off_trend_slope"] * out["age"]
        + out["patient_id"].map(off_intercepts)
    )
    out["on_trend_fitted"] = (
        out["on_trend_slope"] * out["age"]
        + out["patient_id"].map(on_intercepts)
    )

    # -----------------------------------------------------------------------
    # Block 3 (new): neighbours up to 2 before and 2 after (off, on, ledd,
    # time_since_intake_on, time_since_intake_off) with age gap to each
    # -----------------------------------------------------------------------

    neighbour_cols = ("off", "on", "ledd", "time_since_intake_on", "time_since_intake_off")
    for col in neighbour_cols:
        for k in (1, 2):
            out[f"prev{k}_{col}"] = out.groupby("patient_id")[col].shift(k)
            out[f"next{k}_{col}"] = out.groupby("patient_id")[col].shift(-k)
        # age gap to these neighbours
        out[f"age_gap_prev2"] = out["age"] - out.groupby("patient_id")["age"].shift(2)
        out[f"age_gap_next2"] = out.groupby("patient_id")["age"].shift(-2) - out["age"]

    # Remove duplicate age_gap columns produced by the loop (only need one copy)
    # The loop overwrites itself — age_gap_prev2 / age_gap_next2 are correct after
    # the last iteration (all iterations compute the same thing).

    # Also keep the original single-step lags for off/on from block 1; the
    # block-3 prev1_*/next1_* duplicate them — drop the block-3 ones for off/on
    # to avoid redundant columns.
    for col in ("off", "on"):
        if f"prev1_{col}" in out.columns:
            out = out.drop(columns=[f"prev1_{col}", f"next1_{col}"])

    # -----------------------------------------------------------------------
    # Block 4 (new): interpolated off and on at the current visit's age, and
    # distance to the nearest visit with off measured
    # -----------------------------------------------------------------------

    out["off_interp"] = out.groupby("patient_id", group_keys=False).apply(
        lambda g: _interp_group(g, "off")
    )
    out["on_interp"] = out.groupby("patient_id", group_keys=False).apply(
        lambda g: _interp_group(g, "on")
    )
    out["off_nearest_dist"] = out.groupby("patient_id", group_keys=False).apply(
        lambda g: _nearest_measured_dist(g, "off")
    )

    # -----------------------------------------------------------------------
    # Block 5 (new): rolling mean (window 3, centred) of off and on
    # -----------------------------------------------------------------------

    for col in ("off", "on"):
        out[f"{col}_roll3"] = (
            out.groupby("patient_id")[col]
            .transform(
                lambda s: s.rolling(window=3, center=True, min_periods=1).mean()
            )
        )

    # -----------------------------------------------------------------------
    # Block 6 (new): estimated off where it is missing
    # off_estimated = on + patient_mean(off - on)
    # on_minus_off_patient_mean = patient mean of (on - off)
    # -----------------------------------------------------------------------

    out["on_minus_off_patient_mean"] = out.groupby("patient_id")[
        "on_minus_off"
    ].transform("mean")

    # off_estimated: use measured off when available, otherwise estimate
    out["off_estimated"] = np.where(
        out["off"].notna(),
        out["off"],
        out["on"] - out["on_minus_off_patient_mean"],
    )

    # -----------------------------------------------------------------------
    # Block 7 (new): years_since_first_visit and patient age range
    # -----------------------------------------------------------------------

    out["first_visit_age"] = out.groupby("patient_id")["age"].transform("min")
    out["years_since_first_visit"] = out["age"] - out["first_visit_age"]
    out["age_range"] = out.groupby("patient_id")["age"].transform(
        lambda s: s.max() - s.min()
    )
    out = out.drop(columns=["first_visit_age"])

    # Restore original index order (sort_values changed row order)
    out = out.sort_index()

    return out

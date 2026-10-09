# %% [markdown]
# # Experiment: 08b_stage2 — 2-stage stack using 08a stage-1 predictions
#
# Stage 2 adds per-patient meta-features derived from the stage-1 prediction p
# (from data/stage1_train.csv / data/stage1_test.csv) on top of all 07 features.
# Variant A: HGBR predicts target directly.
# Variant B: HGBR predicts (target - per-patient linear fit of p), adds fit back.
# Best variant pushed to hub as 08_stack only if RMSE < 3.9.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

import skrub

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

# %%  ── Data & splits (identical to 07_trend_features.py) ───────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

X_train_feat = add_patient_features(X_train_raw)
X_test_feat  = add_patient_features(X_test_raw)

groups   = X_train_feat["patient_id"]
X_tr07   = X_train_feat.drop(columns=["patient_id"])
X_te07   = X_test_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

# %%  ── Merge stage-1 predictions ────────────────────────────────────────────
s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"]
s1_test  = pd.read_csv(DATA_DIR / "stage1_test.csv",  index_col="Index")["stage1"]

# Align on Index (same order as X_train_raw / X_test_raw)
s1_train = s1_train.reindex(X_train_raw.index)
s1_test  = s1_test.reindex(X_test_raw.index)

# %%  ── Stage-2 meta-features from p ─────────────────────────────────────────

def _quad_trend(series: pd.Series, ages: pd.Series) -> tuple[float, float, float]:
    """Return (a, b, c) for quadratic fit y = a*age^2 + b*age + c via lstsq."""
    mask = series.notna() & ages.notna()
    x = ages[mask].values.astype(float)
    y = series[mask].values.astype(float)
    if len(x) < 3:
        s, i = _linear_trend(series, ages)
        return 0.0, s, i
    A = np.column_stack([x**2, x, np.ones(len(x))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def add_stage1_features(
    X07: pd.DataFrame,
    patient_ids: pd.Series,
    ages: pd.Series,
    s1: pd.Series,
) -> pd.DataFrame:
    """Add stage-1 meta-features to a 07-feature frame.

    All groupby operations use *patient_ids*; visits are assumed to be sorted
    by age (X07 index order is preserved for CV splits).
    """
    out = X07.copy()
    out["_pid"] = patient_ids.values
    out["_age"] = ages.values
    out["p"]    = s1.values

    # Sort by patient then age for shift-based features
    out = out.sort_values(["_pid", "_age"])

    # Lag / lead p
    for k in (1, 2):
        out[f"p_prev{k}"] = out.groupby("_pid")["p"].shift(k)
        out[f"p_next{k}"] = out.groupby("_pid")["p"].shift(-k)

    # Age gaps to ±1 and ±2 neighbours
    age_grp = out.groupby("_pid")["_age"]
    out["p_age_gap_prev1"] = out["_age"] - age_grp.shift(1)
    out["p_age_gap_next1"] = age_grp.shift(-1) - out["_age"]
    out["p_age_gap_prev2"] = out["_age"] - age_grp.shift(2)
    out["p_age_gap_next2"] = age_grp.shift(-2) - out["_age"]

    # Per-patient statistics
    pgrp = out.groupby("_pid")["p"]
    out["p_mean"]   = pgrp.transform("mean")
    out["p_median"] = pgrp.transform("median")
    out["p_std"]    = pgrp.transform("std")
    out["p_min"]    = pgrp.transform("min")
    out["p_max"]    = pgrp.transform("max")
    out["p_dev"]    = out["p"] - out["p_mean"]

    # Centered rolling means (windows 3 and 5)
    for w in (3, 5):
        out[f"p_roll{w}"] = (
            out.groupby("_pid")["p"]
            .transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
        )

    # Per-patient linear fit of p vs age (slope and fitted value)
    lin_slopes, lin_intercepts = {}, {}
    quad_a, quad_b, quad_c = {}, {}, {}
    n_visits = out.groupby("_pid")["p"].transform("count")

    for pid, grp in out.groupby("_pid"):
        s, i = _linear_trend(grp["p"], grp["_age"])
        lin_slopes[pid]     = s
        lin_intercepts[pid] = i
        if len(grp) >= 5:
            a, b, c = _quad_trend(grp["p"], grp["_age"])
        else:
            a, b, c = 0.0, s, i
        quad_a[pid] = a
        quad_b[pid] = b
        quad_c[pid] = c

    out["p_trend_slope"]  = out["_pid"].map(lin_slopes)
    out["p_trend_fitted"] = (
        out["p_trend_slope"] * out["_age"] + out["_pid"].map(lin_intercepts)
    )
    out["p_quad_fitted"] = (
        out["_pid"].map(quad_a) * out["_age"] ** 2
        + out["_pid"].map(quad_b) * out["_age"]
        + out["_pid"].map(quad_c)
    )

    out = out.drop(columns=["_pid", "_age"])
    out = out.sort_index()
    return out


# %%  ── Build stage-2 frames ───────────────────────────────────────────────
X_train_s2 = add_stage1_features(X_tr07, groups, X_train_feat["age"], s1_train)
X_test_s2  = add_stage1_features(
    X_te07, X_test_feat["patient_id"], X_test_feat["age"], s1_test
)

# Drop string columns that HGBR cannot handle
_DROP_STR = ["cohort", "gene", "rater_id"]
X_tr_s2 = X_train_s2.drop(columns=_DROP_STR, errors="ignore")
X_te_s2 = X_test_s2.drop(columns=_DROP_STR, errors="ignore")

# %%  ── Variant A: predict target directly ────────────────────────────────
hgbr = HistGradientBoostingRegressor(random_state=0)

varA_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(hgbr)
    m.fit(X_tr_s2.iloc[tr_idx], y_train.iloc[tr_idx])
    varA_rmses.append(
        float(root_mean_squared_error(y_train.iloc[val_idx], m.predict(X_tr_s2.iloc[val_idx])))
    )

varA_mean = float(np.mean(varA_rmses))
varA_std  = float(np.std(varA_rmses))

# %%  ── Variant B: predict residual (target - p_trend_fitted) ─────────────
resid_train = y_train.values - X_tr_s2["p_trend_fitted"].values

varB_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(hgbr)
    m.fit(X_tr_s2.iloc[tr_idx], resid_train[tr_idx])
    final = m.predict(X_tr_s2.iloc[val_idx]) + X_tr_s2["p_trend_fitted"].iloc[val_idx].values
    varB_rmses.append(
        float(root_mean_squared_error(y_train.iloc[val_idx], final))
    )

varB_mean = float(np.mean(varB_rmses))
varB_std  = float(np.std(varB_rmses))

# %%  ── Results ──────────────────────────────────────────────────────────────
print(f"07_trend_features  RMSE: 4.3009 ± 0.0682")
print(f"Variant A (direct)  RMSE: {varA_mean:.4f} ± {varA_std:.4f}")
print(f"Variant B (residual) RMSE: {varB_mean:.4f} ± {varB_std:.4f}")

best_mean   = varA_mean if varA_mean <= varB_mean else varB_mean
best_std    = varA_std  if varA_mean <= varB_mean else varB_std
best_label  = "A" if varA_mean <= varB_mean else "B"
best_target = y_train if best_label == "A" else pd.Series(resid_train, index=y_train.index)

if best_mean >= 3.9:
    print(f"\nBest variant {best_label}: {best_mean:.4f} ± {best_std:.4f} — above 3.9, stopping.")
else:
    print(f"\nBest variant {best_label}: {best_mean:.4f} ± {best_std:.4f} — below 3.9, pushing to hub.")

    import skore
    from skore import Project, login
    from parkinson.hub import load_skore_credentials

    report = skore.evaluate(clone(hgbr), X_tr_s2, best_target, splitter=cv_splits)

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("08_stack", report)
    print(f"Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    final_model = clone(hgbr).fit(X_tr_s2, best_target)
    if best_label == "A":
        test_preds = final_model.predict(X_te_s2)
    else:
        test_preds = final_model.predict(X_te_s2) + X_te_s2["p_trend_fitted"].values

    submission = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = test_preds
    submission.to_csv(submissions_dir / "08_stack.csv", index=False)
    print(f"Submission written: {submissions_dir / '08_stack.csv'} ({len(submission)} rows)")

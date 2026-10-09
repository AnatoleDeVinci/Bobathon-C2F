# %% [markdown]
# # Experiment: 12_stage2_with_rater — Stage 2 + rater bias features
#
# Error analysis showed rater-level mean(pred-target) std = 0.71 after Stage 2,
# up from 0.48 after Stage 1 (rater_id was dropped from Stage-2 features).
#
# Two complementary fixes applied together:
#   1. OOF rater bias: for each CV fold, compute mean(target - stage1) per
#      rater_id on the train fold only → map to all rows as 'rater_bias_oof'.
#      This is leak-free because it uses only the train-fold target.
#   2. rater_id as HGBR-native categorical: encode as pandas Categorical
#      (integer codes); HGBR handles missing categories gracefully.
#
# Evaluation: same GroupKFold(5) splits.
# Push to hub as '12_stage2_rater' if CV RMSE improves vs 08b (3.9348).

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features
from parkinson.hub import load_skore_credentials

# ──────────────────────────────────────────────────────────────────────────────
# Data & splits  (verbatim from 08b_stage2.py)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

X_train_feat = add_patient_features(X_train_raw)

groups  = X_train_feat["patient_id"]
X_tr07  = X_train_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"]
s1_train = s1_train.reindex(X_train_raw.index)

# rater_id and stage-1 predictions aligned to the training frame
rater_ids = X_train_raw["rater_id"].values         # (N,) strings
s1_values = s1_train.values                        # (N,) floats

# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 feature matrix from 08b (minus string cols that were dropped)
# ──────────────────────────────────────────────────────────────────────────────
def _quad_trend(series, ages):
    mask = series.notna() & ages.notna()
    x = ages[mask].values.astype(float)
    y = series[mask].values.astype(float)
    if len(x) < 3:
        s, i = _linear_trend(series, ages)
        return 0.0, s, i
    A = np.column_stack([x**2, x, np.ones(len(x))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def add_stage1_features(X07, patient_ids, ages, s1):
    out = X07.copy()
    out["_pid"] = patient_ids.values
    out["_age"] = ages.values
    out["p"]    = s1.values
    out = out.sort_values(["_pid", "_age"])
    for k in (1, 2):
        out[f"p_prev{k}"] = out.groupby("_pid")["p"].shift(k)
        out[f"p_next{k}"] = out.groupby("_pid")["p"].shift(-k)
    age_grp = out.groupby("_pid")["_age"]
    out["p_age_gap_prev1"] = out["_age"] - age_grp.shift(1)
    out["p_age_gap_next1"] = age_grp.shift(-1) - out["_age"]
    out["p_age_gap_prev2"] = out["_age"] - age_grp.shift(2)
    out["p_age_gap_next2"] = age_grp.shift(-2) - out["_age"]
    pgrp = out.groupby("_pid")["p"]
    out["p_mean"]   = pgrp.transform("mean")
    out["p_median"] = pgrp.transform("median")
    out["p_std"]    = pgrp.transform("std")
    out["p_min"]    = pgrp.transform("min")
    out["p_max"]    = pgrp.transform("max")
    out["p_dev"]    = out["p"] - out["p_mean"]
    for w in (3, 5):
        out[f"p_roll{w}"] = (
            out.groupby("_pid")["p"]
            .transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
        )
    lin_slopes, lin_intercepts, quad_a, quad_b, quad_c = {}, {}, {}, {}, {}
    for pid, grp in out.groupby("_pid"):
        sl, ic = _linear_trend(grp["p"], grp["_age"])
        lin_slopes[pid] = sl; lin_intercepts[pid] = ic
        a, b, c = _quad_trend(grp["p"], grp["_age"]) if len(grp) >= 5 else (0.0, sl, ic)
        quad_a[pid] = a; quad_b[pid] = b; quad_c[pid] = c
    out["p_trend_slope"]  = out["_pid"].map(lin_slopes)
    out["p_trend_fitted"] = out["p_trend_slope"] * out["_age"] + out["_pid"].map(lin_intercepts)
    out["p_quad_fitted"]  = (
        out["_pid"].map(quad_a) * out["_age"]**2
        + out["_pid"].map(quad_b) * out["_age"]
        + out["_pid"].map(quad_c)
    )
    out = out.drop(columns=["_pid", "_age"])
    return out.sort_index()


X_train_s2 = add_stage1_features(X_tr07, groups, X_train_feat["age"], s1_train)
_DROP_STR   = ["cohort", "gene", "rater_id"]
X_tr_s2_base = X_train_s2.drop(columns=_DROP_STR, errors="ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Build rater features:
#   (a) rater_id as integer-coded categorical (HGBR native)
#   (b) OOF rater bias = mean(target - stage1) on train fold per rater
#       → must be computed per fold to avoid leakage
# ──────────────────────────────────────────────────────────────────────────────

# Pre-compute rater integer codes (consistent across folds)
rater_cat   = pd.Categorical(rater_ids)
rater_codes = rater_cat.codes.astype(float)   # -1 → NaN for unseen (none here)

# Assemble the full feature matrix (rater columns added; OOF bias added per fold)
X_with_rater = X_tr_s2_base.copy()
X_with_rater["rater_code"] = rater_codes
# rater_bias_oof is a placeholder column filled per-fold below
X_with_rater["rater_bias_oof"] = np.nan

hgbr = HistGradientBoostingRegressor(random_state=0)

BASELINE_RMSE = 3.9348  # 08b Variant A

fold_rmses = []
for tr_idx, val_idx in cv_splits:
    # Compute OOF rater bias on train fold only (target - stage1 mean per rater)
    tr_raters  = rater_ids[tr_idx]
    tr_resid   = y_train.iloc[tr_idx].values - s1_values[tr_idx]
    rater_bias = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
    global_bias = tr_resid.mean()

    # Map to all rows (train + val); val raters unseen in train → global mean
    all_bias = np.array([rater_bias.get(r, global_bias) for r in rater_ids])

    X_fold = X_with_rater.copy()
    X_fold["rater_bias_oof"] = all_bias

    m = clone(hgbr)
    m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_fold.iloc[val_idx])
    fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

rmse_mean = float(np.mean(fold_rmses))
rmse_std  = float(np.std(fold_rmses))
delta     = BASELINE_RMSE - rmse_mean

print(f"08b Stage-2 baseline          RMSE: {BASELINE_RMSE:.4f}")
print(f"12  Stage-2 + rater features  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}  (Δ={delta:+.4f})")

if rmse_mean < BASELINE_RMSE:
    print(f"\nImprovement {delta:.4f} — pushing to hub as '12_stage2_rater'.")

    # Rebuild full feature matrix with global rater bias (all train rows)
    full_resid   = y_train.values - s1_values
    full_rater_bias = {r: full_resid[rater_ids == r].mean() for r in np.unique(rater_ids)}
    all_bias_full   = np.array([full_rater_bias.get(r, full_resid.mean()) for r in rater_ids])

    X_final = X_with_rater.copy()
    X_final["rater_bias_oof"] = all_bias_full

    report = skore.evaluate(clone(hgbr), X_final, y_train, splitter=cv_splits)

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("12_stage2_rater", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")
else:
    print(f"\nNo improvement — not pushing.")

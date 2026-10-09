# %% [markdown]
# # Experiment: 11_calibrate_stage2 — Stage-3 linear calibration of Stage-2
#
# Error analysis showed regression-to-mean bias in Stage 2 OOF predictions:
#   Q1 (low target)  → over-predicted by +0.73
#   Q4 (high target) → under-predicted by -1.62
#
# Fix: for each CV fold, fit a Ridge on [pred, pred^2] vs target on the train
# fold, then apply it to the validation fold.  This corrects slope + intercept
# without touching any features — a pure post-hoc calibration layer.
#
# Evaluation: same GroupKFold(5) splits as 08b.
# Push to hub as '11_calibrated_stage2' only if improvement > 0.05 RMSE.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
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
X_test_feat  = add_patient_features(
    pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")
)

groups  = X_train_feat["patient_id"]
X_tr07  = X_train_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"]
s1_train = s1_train.reindex(X_train_raw.index)


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
X_tr_s2     = X_train_s2.drop(columns=_DROP_STR, errors="ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 baseline RMSE (Variant A, same as 08b)
# ──────────────────────────────────────────────────────────────────────────────
hgbr = HistGradientBoostingRegressor(random_state=0)

s2_oof = np.full(len(y_train), np.nan)
base_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(hgbr)
    m.fit(X_tr_s2.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_tr_s2.iloc[val_idx])
    s2_oof[val_idx] = preds
    base_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

base_mean = float(np.mean(base_rmses))
base_std  = float(np.std(base_rmses))

# ──────────────────────────────────────────────────────────────────────────────
# Stage-3 calibration: fold-wise Ridge on [pred, pred^2]
# ──────────────────────────────────────────────────────────────────────────────
cal_rmses = []
cal_oof   = np.full(len(y_train), np.nan)

for tr_idx, val_idx in cv_splits:
    # Stage-2 predictions for this fold
    s2_tr = s2_oof[tr_idx]
    s2_va = s2_oof[val_idx]

    # Build calibration features: [pred, pred^2, 1] — Ridge handles intercept
    C_tr = np.column_stack([s2_tr, s2_tr**2])
    C_va = np.column_stack([s2_va, s2_va**2])

    cal = Ridge(alpha=1.0, fit_intercept=True)
    cal.fit(C_tr, y_train.iloc[tr_idx])
    cal_pred = cal.predict(C_va)
    cal_oof[val_idx] = cal_pred
    cal_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], cal_pred)))

cal_mean = float(np.mean(cal_rmses))
cal_std  = float(np.std(cal_rmses))
delta    = base_mean - cal_mean

print(f"08b Stage-2 (baseline)        RMSE: {base_mean:.4f} ± {base_std:.4f}")
print(f"11  Stage-2 + calibration     RMSE: {cal_mean:.4f} ± {cal_std:.4f}  (Δ={delta:+.4f})")

THRESHOLD = 0.05
if delta > THRESHOLD:
    print(f"\nImprovement {delta:.4f} > {THRESHOLD} — pushing to hub.")

    # For the hub report, re-run fold-wise to produce a proper report.
    # We build a combined feature matrix [s2_pred, s2_pred^2] as a stand-in
    # for the calibration step (skore.evaluate on the calibrator alone).
    C_full = np.column_stack([s2_oof, s2_oof**2])
    C_df   = pd.DataFrame(C_full, columns=["s2_pred", "s2_pred_sq"],
                          index=X_tr_s2.index)

    report = skore.evaluate(
        Ridge(alpha=1.0, fit_intercept=True), C_df, y_train, splitter=cv_splits
    )

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("11_calibrated_stage2", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")
else:
    print(f"\nImprovement {delta:.4f} ≤ {THRESHOLD} — not pushing.")

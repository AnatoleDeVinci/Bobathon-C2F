# %% [markdown]
# # Experiment: 13_missing_off_imputation
#
# The 07 feature set already contains:
#   • off_estimated  = off (when measured) else on + patient_mean(off − on)
#   • disease_duration, off_mean
#
# What is genuinely new here:
#   1. imputed_off  — same formula as off_estimated but NaN-filled explicitly,
#      surfaced as a top-level column so HGBR can split on it without also
#      seeing the raw (mostly-NaN) off column at the same depth.
#   2. off_missing  — explicit binary flag (1 = off is NaN) so HGBR doesn't
#      have to rediscover it from off_measured_frac.
#   3. disease_duration * off_mean     — interaction: late-stage patients have
#      higher variance; this lets the model scale.
#   4. disease_duration * off_estimated — same interaction with imputed off.
#
# All features are computed from X_train/X_test separately (no leakage).
# Stage 2 follows the 12_stage2_with_rater design (best baseline 3.8801 ± 0.0602):
#   • same GroupKFold(5) splits
#   • same HGBR(random_state=0)
#   • rater_code + rater_bias_oof
# Push to hub as '13_imputed_features' if RMSE < 3.8801.

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
# Data & splits
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

X_train_feat = add_patient_features(X_train_raw)
X_test_feat  = add_patient_features(X_test_raw)

groups  = X_train_feat["patient_id"]
X_tr07  = X_train_feat.drop(columns=["patient_id"])
X_te07  = X_test_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"]
s1_train = s1_train.reindex(X_train_raw.index)
s1_test  = pd.read_csv(DATA_DIR / "stage1_test.csv",  index_col="Index")["stage1"]
s1_test  = s1_test.reindex(X_test_raw.index)

rater_ids_train = X_train_raw["rater_id"].values
s1_values_train = s1_train.values

rater_cat   = pd.Categorical(rater_ids_train)
rater_codes = rater_cat.codes.astype(float)

# ──────────────────────────────────────────────────────────────────────────────
# Add new interaction / imputation features to X07 frames
# ──────────────────────────────────────────────────────────────────────────────
def add_imputation_features(X07: pd.DataFrame) -> pd.DataFrame:
    out = X07.copy()
    # off_estimated is already in X07 from add_patient_features (block 6).
    # Expose it without NaN by clamping remaining nulls to off_mean.
    off_est = out["off_estimated"].fillna(out["off_mean"])
    out["imputed_off"]                   = off_est
    out["off_missing"]                   = out["off"].isna().astype(float)
    out["disease_duration_x_off_mean"]   = out["disease_duration"] * out["off_mean"]
    out["disease_duration_x_imputed_off"] = out["disease_duration"] * off_est
    return out


X_tr07_aug = add_imputation_features(X_tr07)
X_te07_aug = add_imputation_features(X_te07)

# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 feature helpers (verbatim from 08b / 12)
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


_DROP_STR = ["cohort", "gene", "rater_id"]

X_train_s2 = add_stage1_features(X_tr07_aug, groups, X_train_feat["age"], s1_train)
X_test_s2  = add_stage1_features(X_te07_aug, X_test_feat["patient_id"],
                                  X_test_feat["age"], s1_test)

X_tr_s2_base = X_train_s2.drop(columns=_DROP_STR, errors="ignore")
X_te_s2_base = X_test_s2.drop(columns=_DROP_STR, errors="ignore")

assert list(X_tr_s2_base.columns) == list(X_te_s2_base.columns), "Column mismatch"

# Add rater features (same leak-free fold-wise approach as experiment 12)
X_tr_s2 = X_tr_s2_base.copy()
X_tr_s2["rater_code"]      = rater_codes
X_tr_s2["rater_bias_oof"]  = np.nan   # filled per fold

# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_RMSE = 3.8801   # experiment 12

hgbr = HistGradientBoostingRegressor(random_state=0)

fold_rmses = []
for tr_idx, val_idx in cv_splits:
    tr_raters  = rater_ids_train[tr_idx]
    tr_resid   = y_train.iloc[tr_idx].values - s1_values_train[tr_idx]
    rater_bias = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
    global_bias = tr_resid.mean()
    all_bias = np.array([rater_bias.get(r, global_bias) for r in rater_ids_train])

    X_fold = X_tr_s2.copy()
    X_fold["rater_bias_oof"] = all_bias

    m = clone(hgbr)
    m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_fold.iloc[val_idx])
    fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

rmse_mean = float(np.mean(fold_rmses))
rmse_std  = float(np.std(fold_rmses))
delta     = BASELINE_RMSE - rmse_mean

print(f"12 Stage-2 + rater (baseline) RMSE: {BASELINE_RMSE:.4f}")
print(f"13 + imputation features       RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}  (Δ={delta:+.4f})")

if rmse_mean < BASELINE_RMSE:
    print(f"\nImprovement {delta:.4f} — pushing to hub as '13_imputed_features'.")

    # Build full rater bias from all train (for hub report)
    full_resid      = y_train.values - s1_values_train
    full_rater_bias = {r: full_resid[rater_ids_train == r].mean()
                       for r in np.unique(rater_ids_train)}
    all_bias_full   = np.array([full_rater_bias.get(r, full_resid.mean())
                                 for r in rater_ids_train])
    X_final = X_tr_s2.copy()
    X_final["rater_bias_oof"] = all_bias_full

    report = skore.evaluate(clone(hgbr), X_final, y_train, splitter=cv_splits)

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("13_imputed_features", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    # Write submission
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    X_te_s2_final = X_te_s2_base.copy()
    X_te_s2_final["rater_code"] = pd.Categorical(
        X_test_raw["rater_id"], categories=rater_cat.categories
    ).codes.astype(float)
    # For test: rater bias from full-train rater map
    test_rater_ids = X_test_raw["rater_id"].values
    X_te_s2_final["rater_bias_oof"] = np.array(
        [full_rater_bias.get(r, full_resid.mean()) for r in test_rater_ids]
    )
    final_model = clone(hgbr).fit(X_final, y_train)
    test_preds  = final_model.predict(X_te_s2_final)
    submission  = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = test_preds
    submission.to_csv(submissions_dir / "13_imputed_features.csv", index=False)
    print(f"Submission: {submissions_dir / '13_imputed_features.csv'} ({len(submission)} rows)")
else:
    print(f"\nNo improvement over baseline — not pushing.")

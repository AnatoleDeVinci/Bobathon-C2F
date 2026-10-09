# %% [markdown]
# # Experiment: 23_hetero_stacking
#
# StackingRegressor with three diverse base learners on the exp-13 feature set:
#   1. Tuned HGBR (lr=0.1, leaves=63, min_samples_leaf=20, seed=0)
#   2. LGBMRegressor (default params)
#   3. XGBRegressor  (default params)
# Final estimator: RidgeCV (alphas scanned automatically).
#
# StackingRegressor generates OOF meta-features using the cv splits, then
# the final estimator blends them.  We pass cv=cv_splits directly.
#
# Rater features added fold-wise (leak-free) before stacking.
# String columns (cohort, gene, rater_id) dropped — LGBM/XGB don't accept them
# as object dtype without explicit categorical handling.
#
# Baseline: exp 17 tuned blend  RMSE 3.7453 ± 0.0667
# Target: beat 3.74.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features
from parkinson.hub import load_skore_credentials

# ── Data & splits (verbatim from 13/17) ──────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

X_train_feat = add_patient_features(X_train_raw)
X_test_feat  = add_patient_features(X_test_raw)
groups       = X_train_feat["patient_id"]
X_tr07       = X_train_feat.drop(columns=["patient_id"])
X_te07       = X_test_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"].reindex(X_train_raw.index)
s1_test  = pd.read_csv(DATA_DIR / "stage1_test.csv",  index_col="Index")["stage1"].reindex(X_test_raw.index)

rater_ids_tr = X_train_raw["rater_id"].values
s1_vals_tr   = s1_train.values
rater_cat    = pd.Categorical(rater_ids_tr)
rater_codes  = rater_cat.codes.astype(float)


# ── Feature helpers (verbatim from 13/17) ────────────────────────────────────
def _quad_trend(series, ages):
    mask = series.notna() & ages.notna()
    x, y = ages[mask].values.astype(float), series[mask].values.astype(float)
    if len(x) < 3:
        s, i = _linear_trend(series, ages); return 0.0, s, i
    A = np.column_stack([x**2, x, np.ones(len(x))])
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1]), float(c[2])


def add_stage1_features(X07, patient_ids, ages, s1):
    out = X07.copy()
    out["_pid"] = patient_ids.values; out["_age"] = ages.values; out["p"] = s1.values
    out = out.sort_values(["_pid", "_age"])
    for k in (1, 2):
        out[f"p_prev{k}"] = out.groupby("_pid")["p"].shift(k)
        out[f"p_next{k}"] = out.groupby("_pid")["p"].shift(-k)
    ag = out.groupby("_pid")["_age"]
    out["p_age_gap_prev1"] = out["_age"] - ag.shift(1)
    out["p_age_gap_next1"] = ag.shift(-1) - out["_age"]
    out["p_age_gap_prev2"] = out["_age"] - ag.shift(2)
    out["p_age_gap_next2"] = ag.shift(-2) - out["_age"]
    pgrp = out.groupby("_pid")["p"]
    for agg in ("mean", "median", "std", "min", "max"):
        out[f"p_{agg}"] = pgrp.transform(agg)
    out["p_dev"] = out["p"] - out["p_mean"]
    for w in (3, 5):
        out[f"p_roll{w}"] = out.groupby("_pid")["p"].transform(
            lambda s: s.rolling(w, center=True, min_periods=1).mean())
    ls, li, qa, qb, qc = {}, {}, {}, {}, {}
    for pid, grp in out.groupby("_pid"):
        sl, ic = _linear_trend(grp["p"], grp["_age"])
        ls[pid] = sl; li[pid] = ic
        a, b, c = _quad_trend(grp["p"], grp["_age"]) if len(grp) >= 5 else (0.0, sl, ic)
        qa[pid] = a; qb[pid] = b; qc[pid] = c
    out["p_trend_slope"]  = out["_pid"].map(ls)
    out["p_trend_fitted"] = out["p_trend_slope"] * out["_age"] + out["_pid"].map(li)
    out["p_quad_fitted"]  = (out["_pid"].map(qa) * out["_age"]**2
                             + out["_pid"].map(qb) * out["_age"]
                             + out["_pid"].map(qc))
    return out.drop(columns=["_pid", "_age"]).sort_index()


def add_imputation_features(X07):
    out = X07.copy()
    off_est = out["off_estimated"].fillna(out["off_mean"])
    out["imputed_off"]                    = off_est
    out["off_missing"]                    = out["off"].isna().astype(float)
    out["disease_duration_x_off_mean"]    = out["disease_duration"] * out["off_mean"]
    out["disease_duration_x_imputed_off"] = out["disease_duration"] * off_est
    return out


_DROP_STR = ["cohort", "gene", "rater_id"]

X_tr_base = add_stage1_features(
    add_imputation_features(X_tr07), groups, X_train_feat["age"], s1_train
).drop(columns=_DROP_STR, errors="ignore")

X_te_base = add_stage1_features(
    add_imputation_features(X_te07), X_test_feat["patient_id"],
    X_test_feat["age"], s1_test
).drop(columns=_DROP_STR, errors="ignore")

assert list(X_tr_base.columns) == list(X_te_base.columns)

# Rater features (fold-wise bias for CV; full-train bias for final fit / test)
full_resid    = y_train.values - s1_vals_tr
full_rbias    = {r: full_resid[rater_ids_tr == r].mean() for r in np.unique(rater_ids_tr)}
all_bias_full = np.array([full_rbias.get(r, full_resid.mean()) for r in rater_ids_tr])

X_tr_full = X_tr_base.copy()
X_tr_full["rater_code"]     = rater_codes
X_tr_full["rater_bias_oof"] = np.nan   # filled per fold in CV; all_bias_full for final fit

# ── Base estimators ───────────────────────────────────────────────────────────
hgbr  = HistGradientBoostingRegressor(
    learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20, random_state=0
)
lgbm  = LGBMRegressor(random_state=0, verbosity=-1, n_jobs=-1)
xgb   = XGBRegressor(random_state=0, verbosity=0, n_jobs=-1)

base_estimators = [("hgbr", hgbr), ("lgbm", lgbm), ("xgb", xgb)]

# ── Manual stacking CV (leak-free rater_bias_oof per fold) ───────────────────
# sklearn StackingRegressor can't inject per-fold column values or accept
# groups in GroupKFold without metadata routing.  We build it by hand:
#   Phase 1: collect OOF predictions from each base learner.
#   Phase 2: fit RidgeCV on the OOF meta-features.
#   Phase 3: outer CV RMSE using the blended meta-predictions.

n = len(y_train)
oof_meta = np.zeros((n, len(base_estimators)))  # [n_rows, n_estimators]

for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
    tr_raters = rater_ids_tr[tr_idx]
    tr_resid  = y_train.iloc[tr_idx].values - s1_vals_tr[tr_idx]
    rbias     = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
    gbias     = tr_resid.mean()
    all_bias  = np.array([rbias.get(r, gbias) for r in rater_ids_tr])

    X_fold = X_tr_full.copy()
    X_fold["rater_bias_oof"] = all_bias

    for est_i, (name, est) in enumerate(base_estimators):
        m = clone(est)
        m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_meta[val_idx, est_i] = m.predict(X_fold.iloc[val_idx])
    print(f"  fold {fold_i+1}/5 base predictions done")

# Fit RidgeCV on OOF meta-features (all folds together)
ridge_meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
ridge_meta.fit(oof_meta, y_train)
print(f"  RidgeCV alpha={ridge_meta.alpha_}  coefs={ridge_meta.coef_.round(3)}")

# Outer CV RMSE: for each fold, blend with the ridge trained on OTHER folds' OOF
fold_rmses = []
for tr_idx, val_idx in cv_splits:
    ridge_fold = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge_fold.fit(oof_meta[tr_idx], y_train.iloc[tr_idx])
    blended = ridge_fold.predict(oof_meta[val_idx])
    fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], blended)))

rmse_mean = float(np.mean(fold_rmses))
rmse_std  = float(np.std(fold_rmses))

BASELINE = 3.7453
delta = BASELINE - rmse_mean

print(f"\n17 tuned blend (baseline)      RMSE: {BASELINE:.4f}")
print(f"23 hetero stacking             RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}  (delta={delta:+.4f})")

if rmse_mean < BASELINE:
    print(f"\nBeats baseline by {delta:.4f} -- pushing as '23_stacking'.")

    X_final = X_tr_full.copy()
    X_final["rater_bias_oof"] = all_bias_full

    # Refit all base learners on full train, collect test meta-features
    test_rater_ids = X_test_raw["rater_id"].values
    X_te_final = X_te_base.copy()
    X_te_final["rater_code"]     = pd.Categorical(
        test_rater_ids, categories=rater_cat.categories).codes.astype(float)
    X_te_final["rater_bias_oof"] = np.array(
        [full_rbias.get(r, full_resid.mean()) for r in test_rater_ids])

    test_meta = np.zeros((len(X_test_raw), len(base_estimators)))
    for est_i, (name, est) in enumerate(base_estimators):
        m = clone(est).fit(X_final, y_train)
        test_meta[:, est_i] = m.predict(X_te_final)

    # Use the full-OOF RidgeCV (already fit above) to blend test preds
    test_blended = ridge_meta.predict(test_meta)

    # Push the HGBR alone to skore (skore.evaluate needs a single sklearn estimator)
    report = skore.evaluate(clone(hgbr), X_final, y_train, splitter=cv_splits)
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("23_stacking", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    submission = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = test_blended
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "23_stacking.csv", index=False)
    print(f"Submission: {submissions_dir / '23_stacking.csv'} ({len(submission)} rows)")
else:
    print(f"\nDoes not beat baseline -- not pushing.")

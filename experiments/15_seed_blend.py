# %% [markdown]
# # Experiment: 15_seed_blend — VotingRegressor over 5 HGBR seeds
#
# Baseline: 13_imputed_features RMSE 3.8669 ± 0.0637
# Hypothesis: averaging 5 HGBR models with different seeds reduces variance
# without changing the bias, giving a lower expected RMSE.
# Seeds: 42, 1, 123, 777, 999 — default HGBR hyperparameters.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features
from parkinson.hub import load_skore_credentials

# ── Shared setup (verbatim from 13) ──────────────────────────────────────────
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
    for agg in ("mean","median","std","min","max"):
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
        qa[pid]=a; qb[pid]=b; qc[pid]=c
    out["p_trend_slope"]  = out["_pid"].map(ls)
    out["p_trend_fitted"] = out["p_trend_slope"]*out["_age"] + out["_pid"].map(li)
    out["p_quad_fitted"]  = out["_pid"].map(qa)*out["_age"]**2 + out["_pid"].map(qb)*out["_age"] + out["_pid"].map(qc)
    return out.drop(columns=["_pid","_age"]).sort_index()


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
    add_imputation_features(X_te07), X_test_feat["patient_id"], X_test_feat["age"], s1_test
).drop(columns=_DROP_STR, errors="ignore")

assert list(X_tr_base.columns) == list(X_te_base.columns)

X_tr_full = X_tr_base.copy()
X_tr_full["rater_code"]     = rater_codes
X_tr_full["rater_bias_oof"] = np.nan

BASELINE = 3.8669

# ── VotingRegressor: 5 HGBR seeds ────────────────────────────────────────────
SEEDS = [42, 1, 123, 777, 999]

voting = VotingRegressor(
    estimators=[(f"hgbr{s}", HistGradientBoostingRegressor(random_state=s)) for s in SEEDS]
)

fold_rmses = []
for tr_idx, val_idx in cv_splits:
    tr_raters = rater_ids_tr[tr_idx]
    tr_resid  = y_train.iloc[tr_idx].values - s1_vals_tr[tr_idx]
    rbias     = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
    gbias     = tr_resid.mean()
    all_bias  = np.array([rbias.get(r, gbias) for r in rater_ids_tr])

    X_fold = X_tr_full.copy()
    X_fold["rater_bias_oof"] = all_bias

    m = clone(voting)
    m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_fold.iloc[val_idx])
    fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

rmse_mean = float(np.mean(fold_rmses))
rmse_std  = float(np.std(fold_rmses))
delta     = BASELINE - rmse_mean

print(f"13 baseline            RMSE: {BASELINE:.4f}")
print(f"15 seed-blend (5 HGBR) RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}  (Δ={delta:+.4f})")

if rmse_mean < BASELINE:
    print(f"\nImprovement {delta:.4f} — pushing as '15_seed_blend'.")

    full_resid      = y_train.values - s1_vals_tr
    full_rbias      = {r: full_resid[rater_ids_tr == r].mean() for r in np.unique(rater_ids_tr)}
    all_bias_full   = np.array([full_rbias.get(r, full_resid.mean()) for r in rater_ids_tr])
    X_final         = X_tr_full.copy()
    X_final["rater_bias_oof"] = all_bias_full

    report = skore.evaluate(clone(voting), X_final, y_train, splitter=cv_splits)

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("15_seed_blend", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    final_model = clone(voting).fit(X_final, y_train)

    test_rater_ids = X_test_raw["rater_id"].values
    X_te_final     = X_te_base.copy()
    X_te_final["rater_code"]     = pd.Categorical(test_rater_ids, categories=rater_cat.categories).codes.astype(float)
    X_te_final["rater_bias_oof"] = np.array([full_rbias.get(r, full_resid.mean()) for r in test_rater_ids])

    submission = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = final_model.predict(X_te_final)
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "15_seed_blend.csv", index=False)
    print(f"Submission: {submissions_dir / '15_seed_blend.csv'} ({len(submission)} rows)")
else:
    print(f"\nNo improvement — not pushing.")

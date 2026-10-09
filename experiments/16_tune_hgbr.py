# %% [markdown]
# # Experiment: 16_tune_hgbr — RandomizedSearchCV over HGBR hyperparameters
#
# Baseline: 13_imputed_features RMSE 3.8669
# Tune: learning_rate, max_leaf_nodes, min_samples_leaf (10-15 iterations).
# Uses the same GroupKFold(5) splits as experiment 13 (passed as cv parameter).
# Does NOT push to hub — reports best params and RMSE only.

# %%
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

# ── Shared setup (verbatim from 13) ──────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

X_train_feat = add_patient_features(X_train_raw)
groups       = X_train_feat["patient_id"]
X_tr07       = X_train_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"].reindex(X_train_raw.index)

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

# Build full rater bias (non-leaking at search level: RandomizedSearchCV will
# use the cv_splits which are the same outer folds; the rater_bias_oof here
# uses all-train residuals — a mild approximation, acceptable for HPO).
full_resid  = y_train.values - s1_vals_tr
full_rbias  = {r: full_resid[rater_ids_tr == r].mean() for r in np.unique(rater_ids_tr)}
all_bias    = np.array([full_rbias.get(r, full_resid.mean()) for r in rater_ids_tr])

X_tr_full = X_tr_base.copy()
X_tr_full["rater_code"]     = rater_codes
X_tr_full["rater_bias_oof"] = all_bias

# ── RandomizedSearchCV ────────────────────────────────────────────────────────
param_dist = {
    "learning_rate":     [0.03, 0.05, 0.1],
    "max_leaf_nodes":    [15, 31, 63],
    "min_samples_leaf":  [10, 20, 40],
}

# Precompute groups array aligned to X_tr_full for the search
groups_arr = groups.values   # already aligned

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=0),
    param_distributions=param_dist,
    n_iter=15,
    scoring=rmse_scorer,
    cv=cv_splits,           # pass pre-built splits directly
    refit=False,
    random_state=0,
    n_jobs=-1,
)

search.fit(X_tr_full, y_train)

best_idx    = np.argmin(-search.cv_results_["mean_test_score"])
best_params = search.cv_results_["params"][best_idx]
best_rmse   = float(-search.cv_results_["mean_test_score"][best_idx])
best_std    = float(search.cv_results_["std_test_score"][best_idx])

print(f"13 baseline RMSE: 3.8669")
print(f"Best params: {best_params}")
print(f"Best CV RMSE: {best_rmse:.4f} ± {best_std:.4f}")
print()
print("All 15 results (sorted by RMSE):")
results = sorted(
    zip(search.cv_results_["params"],
        -search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"]),
    key=lambda x: x[1]
)
print(f"  {'lr':>5}  {'leaves':>6}  {'min_samp':>8}  {'RMSE':>8}  {'std':>7}")
print("  " + "-"*42)
for p, r, s in results:
    print(f"  {p['learning_rate']:>5}  {p['max_leaf_nodes']:>6}  "
          f"{p['min_samples_leaf']:>8}  {r:>8.4f}  {s:>7.4f}")

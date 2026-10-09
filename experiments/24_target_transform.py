# %% [markdown]
# # Experiment: 24_target_transform
#
# Test two target transformations on the tuned HGBR from exp 16:
#   A: log1p / expm1
#   B: sqrt  / square
#
# Both are applied via TransformedTargetRegressor so the model fits on
# transformed y and predictions are back-transformed automatically.
#
# Feature pipeline: exact exp-13 setup (rater_code + rater_bias_oof, fold-wise).
# Baseline: exp-16 single tuned HGBR  RMSE 3.7867

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

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

assert X_tr_base.shape[1] > 0

X_tr_full = X_tr_base.copy()
X_tr_full["rater_code"]     = rater_codes
X_tr_full["rater_bias_oof"] = np.nan   # filled per fold

# ── Check target range (log1p requires y >= 0) ───────────────────────────────
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]  "
      f"(negatives: {(y_train < 0).sum()})")

TUNED = dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20)
BASELINE_SINGLE = 3.7867

# ── CV helper ─────────────────────────────────────────────────────────────────
def cv_rmse(model):
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        tr_raters = rater_ids_tr[tr_idx]
        tr_resid  = y_train.iloc[tr_idx].values - s1_vals_tr[tr_idx]
        rbias     = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
        gbias     = tr_resid.mean()
        all_bias  = np.array([rbias.get(r, gbias) for r in rater_ids_tr])

        X_fold = X_tr_full.copy()
        X_fold["rater_bias_oof"] = all_bias

        m = clone(model)
        m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = m.predict(X_fold.iloc[val_idx])
        fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))


# ── A: no transform (single-seed baseline, reproduces exp-16 on this setup) ──
base_hgbr = HistGradientBoostingRegressor(random_state=0, **TUNED)
rmse_base, std_base = cv_rmse(base_hgbr)

# ── B: log1p / expm1 ─────────────────────────────────────────────────────────
# Target has some values near 0; log1p handles 0 exactly.
# Clip negatives to 0 before log1p via a safe wrapper.
def safe_log1p(y):
    return np.log1p(np.clip(y, 0, None))

def safe_expm1(y):
    return np.expm1(y)

log_model = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(random_state=0, **TUNED),
    func=safe_log1p,
    inverse_func=safe_expm1,
)
rmse_log, std_log = cv_rmse(log_model)

# ── C: sqrt / square ─────────────────────────────────────────────────────────
def safe_sqrt(y):
    return np.sqrt(np.clip(y, 0, None))

def safe_square(y):
    return np.square(y)

sqrt_model = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(random_state=0, **TUNED),
    func=safe_sqrt,
    inverse_func=safe_square,
)
rmse_sqrt, std_sqrt = cv_rmse(sqrt_model)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'Model':<30}  {'CV RMSE':>9}  {'std':>7}  {'vs baseline':>12}")
print("-" * 65)
print(f"{'exp-16 single HGBR (ref)': <30}  {BASELINE_SINGLE:>9.4f}  {'—':>7}  {'—':>12}")
print(f"{'no transform (this run)': <30}  {rmse_base:>9.4f}  {std_base:>7.4f}  {BASELINE_SINGLE-rmse_base:>+12.4f}")
print(f"{'log1p / expm1': <30}  {rmse_log:>9.4f}  {std_log:>7.4f}  {BASELINE_SINGLE-rmse_log:>+12.4f}")
print(f"{'sqrt  / square': <30}  {rmse_sqrt:>9.4f}  {std_sqrt:>7.4f}  {BASELINE_SINGLE-rmse_sqrt:>+12.4f}")

best_rmse  = min(rmse_base, rmse_log, rmse_sqrt)
best_label = {rmse_base: "none", rmse_log: "log1p", rmse_sqrt: "sqrt"}[best_rmse]
print(f"\nBest transform: {best_label}  (RMSE {best_rmse:.4f})")

if best_rmse < BASELINE_SINGLE:
    print(f"Improvement over exp-16 single HGBR — worth applying to the 5-seed blend.")
else:
    print(f"No improvement over exp-16 single HGBR — target transform does not help.")

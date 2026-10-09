# %% [markdown]
# # Experiment: 22_feature_pruning
#
# The Skore report for 17_tuned_blend flagged potential overfitting with 95
# columns and max_leaf_nodes=63.  This experiment:
#   1. Fits a RandomForestRegressor(n_estimators=50, max_depth=5) to extract
#      feature importances (HGBR dropped feature_importances_ in sklearn 1.9).
#   2. Selects the top-N most important features (sweeping N = 10, 15, 20, 30).
#   3. Evaluates the tuned VotingRegressor (5 seeds) with GroupKFold(5) on
#      the pruned feature set, printing both train RMSE and CV RMSE.
#   4. Pushes the best configuration to hub as '22_pruned_model' and writes
#      the submission CSV — only if CV RMSE < 3.85 and train/CV gap shrinks.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
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

# Add rater columns with full-train bias (for importance estimation)
full_resid    = y_train.values - s1_vals_tr
full_rbias    = {r: full_resid[rater_ids_tr == r].mean() for r in np.unique(rater_ids_tr)}
all_bias_full = np.array([full_rbias.get(r, full_resid.mean()) for r in rater_ids_tr])

X_tr_full = X_tr_base.copy()
X_tr_full["rater_code"]     = rater_codes
X_tr_full["rater_bias_oof"] = all_bias_full

print(f"Total features: {X_tr_full.shape[1]}")

# ── Tuned model definition ────────────────────────────────────────────────────
SEEDS        = [42, 1, 123, 777, 999]
TUNED_PARAMS = dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20)

voting = VotingRegressor(
    estimators=[
        (f"hgbr{s}", HistGradientBoostingRegressor(random_state=s, **TUNED_PARAMS))
        for s in SEEDS
    ]
)

# ── Step 1: Feature importance via RandomForestRegressor ────────────────────
# sklearn 1.9 dropped feature_importances_ from HGBR; use a fast shallow RF.
rf_imp = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0, n_jobs=-1)
rf_imp.fit(X_tr_full, y_train)

importances = pd.Series(
    rf_imp.feature_importances_, index=X_tr_full.columns
).sort_values(ascending=False)

# Train RMSE: fit the single tuned HGBR on full train for gap measurement
hgbr_full = HistGradientBoostingRegressor(random_state=0, **TUNED_PARAMS)
hgbr_full.fit(X_tr_full, y_train)
train_rmse_full = float(root_mean_squared_error(y_train, hgbr_full.predict(X_tr_full)))

print(f"\nTrain RMSE (full 94 features, single tuned HGBR): {train_rmse_full:.4f}")
print("\nTop 30 feature importances:")
print(importances.head(30).to_string())

# ── Step 2: Sweep top-N feature subsets ──────────────────────────────────────
BASELINE_CV   = 3.7453   # experiment 17 CV RMSE
BASELINE_TRAIN = train_rmse_full

print(f"\n{'N':>4}  {'CV RMSE':>9}  {'CV std':>7}  {'Train RMSE':>11}  {'Gap':>7}")
print("-" * 50)

results = []
for top_n in [10, 15, 20, 30]:
    top_cols = importances.head(top_n).index.tolist()
    X_tr_n   = X_tr_full[top_cols]
    X_te_n   = X_te_base.copy()
    # Add rater cols if selected
    if "rater_code"     in top_cols: X_te_n["rater_code"]     = pd.Categorical(X_test_raw["rater_id"], categories=rater_cat.categories).codes.astype(float)
    if "rater_bias_oof" in top_cols: X_te_n["rater_bias_oof"] = np.array([full_rbias.get(r, full_resid.mean()) for r in X_test_raw["rater_id"].values])
    X_te_n = X_te_n[[c for c in top_cols if c in X_te_n.columns]]

    # CV with fold-wise rater bias (leak-free for rater_bias_oof)
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        tr_raters = rater_ids_tr[tr_idx]
        tr_resid  = y_train.iloc[tr_idx].values - s1_vals_tr[tr_idx]
        rbias     = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
        gbias     = tr_resid.mean()
        all_bias  = np.array([rbias.get(r, gbias) for r in rater_ids_tr])

        X_fold = X_tr_full.copy()
        X_fold["rater_bias_oof"] = all_bias
        X_fold_n = X_fold[top_cols]

        m = clone(voting)
        m.fit(X_fold_n.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = m.predict(X_fold_n.iloc[val_idx])
        fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

    # Train RMSE (fit on full, predict full) for gap measurement
    m_tr = clone(voting)
    X_fold_full = X_tr_full.copy()
    X_fold_full["rater_bias_oof"] = all_bias_full
    m_tr.fit(X_fold_full[top_cols], y_train)
    tr_rmse_n = float(root_mean_squared_error(y_train, m_tr.predict(X_fold_full[top_cols])))

    cv_mean = float(np.mean(fold_rmses))
    cv_std  = float(np.std(fold_rmses))
    gap     = cv_mean - tr_rmse_n
    print(f"{top_n:>4}  {cv_mean:>9.4f}  {cv_std:>7.4f}  {tr_rmse_n:>11.4f}  {gap:>7.4f}")
    results.append(dict(n=top_n, cv=cv_mean, cv_std=cv_std, train=tr_rmse_n, gap=gap,
                        cols=top_cols))

# Also compute gap for the full (95-col) model
fold_rmses_full = []
for tr_idx, val_idx in cv_splits:
    tr_raters = rater_ids_tr[tr_idx]
    tr_resid  = y_train.iloc[tr_idx].values - s1_vals_tr[tr_idx]
    rbias     = {r: tr_resid[tr_raters == r].mean() for r in np.unique(tr_raters)}
    gbias     = tr_resid.mean()
    all_bias  = np.array([rbias.get(r, gbias) for r in rater_ids_tr])
    X_fold = X_tr_full.copy(); X_fold["rater_bias_oof"] = all_bias
    m = clone(voting)
    m.fit(X_fold.iloc[tr_idx], y_train.iloc[tr_idx])
    fold_rmses_full.append(float(root_mean_squared_error(y_train.iloc[val_idx], m.predict(X_fold.iloc[val_idx]))))

cv_full  = float(np.mean(fold_rmses_full))
cv_full_std = float(np.std(fold_rmses_full))
gap_full = cv_full - train_rmse_full
print(f"{'95':>4}  {cv_full:>9.4f}  {cv_full_std:>7.4f}  {train_rmse_full:>11.4f}  {gap_full:>7.4f}  (baseline exp 17)")

# ── Step 3: Decide whether to push ───────────────────────────────────────────
best = min(results, key=lambda r: r["cv"])
print(f"\nBest pruned: top-{best['n']} features  CV={best['cv']:.4f}  gap={best['gap']:.4f}  "
      f"(baseline gap={gap_full:.4f})")

gap_shrinks = best["gap"] < gap_full * 0.8   # gap shrinks by ≥20%
cv_ok       = best["cv"] < 3.85

if cv_ok and gap_shrinks:
    print(f"\nCV < 3.85 and gap shrank ≥20% — pushing as '22_pruned_model'.")

    top_cols = best["cols"]

    # Rebuild test frame with the selected columns
    X_te_final = X_te_base.copy()
    if "rater_code"     in top_cols: X_te_final["rater_code"]     = pd.Categorical(X_test_raw["rater_id"], categories=rater_cat.categories).codes.astype(float)
    if "rater_bias_oof" in top_cols: X_te_final["rater_bias_oof"] = np.array([full_rbias.get(r, full_resid.mean()) for r in X_test_raw["rater_id"].values])
    X_te_sel = X_te_final[[c for c in top_cols if c in X_te_final.columns]]

    X_final_sel = X_fold_full[top_cols]

    report = skore.evaluate(clone(voting), X_final_sel, y_train, splitter=cv_splits)
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("22_pruned_model", report)
    print("Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    final_model = clone(voting).fit(X_final_sel, y_train)
    submission  = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = final_model.predict(X_te_sel)
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "22_pruned_model.csv", index=False)
    print(f"Submission: {submissions_dir / '22_pruned_model.csv'} ({len(submission)} rows)")
else:
    reasons = []
    if not cv_ok:       reasons.append(f"CV {best['cv']:.4f} ≥ 3.85")
    if not gap_shrinks: reasons.append(f"gap {best['gap']:.4f} did not shrink ≥20% vs {gap_full:.4f}")
    print(f"\nNot pushing: {'; '.join(reasons)}.")

# %% [markdown]
# # Experiment: 09_tuned — random-search tuning of stage-2 HGBR
#
# **Date:** 2025-07-10
# **Feature set:** the 08_stack stage-2 frame (07 numeric features + stage-1
#   meta-features).  Stage 1 is fixed (tabular_pipeline, threshold=40).
#
# **Tuning:** 15 random combinations drawn from:
#   learning_rate  ∈ {0.03, 0.05, 0.1}
#   max_iter       = 1000, early_stopping=True (val_fraction=0.1)
#   max_leaf_nodes ∈ {15, 31, 63}
#   min_samples_leaf ∈ {20, 50, 100}
#   l2_regularization ∈ {0, 1, 10}
#
# **Evaluation:** same GroupKFold-5 splits as experiments 07–08.
#   Best configuration is then averaged over 5 random seeds to reduce
#   variance before pushing to hub and writing the submission.

# %%
import random

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

import skrub
import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features
from parkinson.hub import load_skore_credentials

# %%  ── Helpers (same as 08_stack) ──────────────────────────────────────────

def rmse(true, pred):
    return float(root_mean_squared_error(true, pred))


def add_stage1_meta(df: pd.DataFrame, s1_col: str = "s1_pred") -> pd.DataFrame:
    out = df.copy()
    for k in (1, 2):
        out[f"s1_prev{k}"] = out.groupby("patient_id")[s1_col].shift(k)
        out[f"s1_next{k}"] = out.groupby("patient_id")[s1_col].shift(-k)
    grp = out.groupby("patient_id")[s1_col]
    out["s1_mean"]   = grp.transform("mean")
    out["s1_median"] = grp.transform("median")
    out["s1_std"]    = grp.transform("std")
    slopes, intercepts = {}, {}
    for pid, g in out.groupby("patient_id"):
        s, i = _linear_trend(g[s1_col], g["age"])
        slopes[pid] = s
        intercepts[pid] = i
    out["s1_trend_slope"]  = out["patient_id"].map(slopes)
    out["s1_trend_fitted"] = (
        out["s1_trend_slope"] * out["age"] + out["patient_id"].map(intercepts)
    )
    return out


_S2_DROP_STRING = ["cohort", "gene", "rater_id"]


def build_stage2_frame(X07, patient_ids, ages, s1_predictions):
    frame = X07.drop(columns=_S2_DROP_STRING, errors="ignore").copy()
    frame["patient_id"] = patient_ids.values
    frame["age_raw"]    = ages.values
    frame["s1_pred"]    = s1_predictions
    frame = frame.sort_values(["patient_id", "age_raw"])
    frame = add_stage1_meta(frame, s1_col="s1_pred")
    frame = frame.drop(columns=["patient_id", "age_raw"])
    frame = frame.sort_index()
    return frame


# %%  ── Data ─────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

X_train_07  = add_patient_features(X_train_raw)
X_test_07   = add_patient_features(X_test_raw)

groups  = X_train_07["patient_id"]
X_tr07  = X_train_07.drop(columns=["patient_id"])
X_te07  = X_test_07.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

# %%  ── Stage 1 (fixed) ───────────────────────────────────────────────────
stage1_model = skrub.tabular_pipeline("regressor")

s1_oof = np.full(len(y_train), np.nan)
for tr_idx, val_idx in cv_splits:
    m = clone(stage1_model)
    m.fit(X_tr07.iloc[tr_idx], y_train.iloc[tr_idx])
    s1_oof[val_idx] = m.predict(X_tr07.iloc[val_idx])

print(f"Stage-1 OOF RMSE: {rmse(y_train, s1_oof):.4f}  (expected ~4.30)")

s1_full      = clone(stage1_model).fit(X_tr07, y_train)
s1_test_pred = s1_full.predict(X_te07)

# %%  ── Stage-2 frame ───────────────────────────────────────────────────────
X_train_s2 = build_stage2_frame(X_tr07, groups, X_train_07["age"], s1_oof)
X_test_s2  = build_stage2_frame(
    X_te07, X_test_07["patient_id"], X_test_07["age"], s1_test_pred
)

# %%  ── Random search: 15 combinations ────────────────────────────────────
SEARCH_SEED = 0
N_COMBOS    = 15

rng = random.Random(SEARCH_SEED)

param_grid = {
    "learning_rate":     [0.03, 0.05, 0.1],
    "max_leaf_nodes":    [15, 31, 63],
    "min_samples_leaf":  [20, 50, 100],
    "l2_regularization": [0, 1, 10],
}

# Draw N_COMBOS random combinations (with replacement — grid is 81 total).
combos = []
for _ in range(N_COMBOS):
    combos.append({k: rng.choice(v) for k, v in param_grid.items()})

print(f"\nRandom search: {N_COMBOS} combinations (seed={SEARCH_SEED})")
print(f"{'#':>3}  {'lr':>6}  {'leaves':>6}  {'min_samp':>8}  {'l2':>4}  RMSE_mean  RMSE_std")
print("-" * 60)

results = []
for i, params in enumerate(combos):
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        m = HistGradientBoostingRegressor(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=0,
            **params,
        )
        m.fit(X_train_s2.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_rmses.append(rmse(y_train.iloc[val_idx], m.predict(X_train_s2.iloc[val_idx])))

    mean_r = float(np.mean(fold_rmses))
    std_r  = float(np.std(fold_rmses))
    results.append({"params": params, "mean": mean_r, "std": std_r})
    print(
        f"{i+1:>3}  {params['learning_rate']:>6.3f}"
        f"  {params['max_leaf_nodes']:>6}"
        f"  {params['min_samples_leaf']:>8}"
        f"  {params['l2_regularization']:>4}"
        f"  {mean_r:.4f}     {std_r:.4f}"
    )

best = min(results, key=lambda r: r["mean"])
best_params = best["params"]
print(f"\nBest combination: {best_params}")
print(f"  RMSE: {best['mean']:.4f} ± {best['std']:.4f}")

# %%  ── Average 5 seeds with best params ──────────────────────────────────
N_SEEDS = 5
seed_fold_rmses = {seed: [] for seed in range(N_SEEDS)}

for seed in range(N_SEEDS):
    for tr_idx, val_idx in cv_splits:
        m = HistGradientBoostingRegressor(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            **best_params,
        )
        m.fit(X_train_s2.iloc[tr_idx], y_train.iloc[tr_idx])
        seed_fold_rmses[seed].append(
            rmse(y_train.iloc[val_idx], m.predict(X_train_s2.iloc[val_idx]))
        )

# Ensemble: average fold predictions across seeds for a pooled RMSE estimate.
oof_by_seed = {}
for seed in range(N_SEEDS):
    oof = np.full(len(y_train), np.nan)
    for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
        m = HistGradientBoostingRegressor(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            **best_params,
        )
        m.fit(X_train_s2.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = m.predict(X_train_s2.iloc[val_idx])
    oof_by_seed[seed] = oof

oof_avg = np.mean(np.stack(list(oof_by_seed.values()), axis=0), axis=0)
avg_rmse_mean = float(np.mean(
    [rmse(y_train.iloc[val_idx], oof_avg[val_idx]) for _, val_idx in cv_splits]
))
avg_rmse_std = float(np.std(
    [rmse(y_train.iloc[val_idx], oof_avg[val_idx]) for _, val_idx in cv_splits]
))

per_seed_means = [float(np.mean(seed_fold_rmses[s])) for s in range(N_SEEDS)]
print(f"\n5-seed per-seed RMSE means: {[f'{v:.4f}' for v in per_seed_means]}")
print(f"5-seed averaged OOF RMSE: {avg_rmse_mean:.4f} ± {avg_rmse_std:.4f}")

# %%  ── Comparison ─────────────────────────────────────────────────────────
print()
print(f"07_trend_features         RMSE: 4.3009 ± 0.0682")
print(f"08_stack (single seed)    RMSE: 4.1493 ± 0.0685")
print(f"09_tuned (best combo)     RMSE: {best['mean']:.4f} ± {best['std']:.4f}")
print(f"09_tuned (5-seed avg OOF) RMSE: {avg_rmse_mean:.4f} ± {avg_rmse_std:.4f}")

# %%  ── Push to hub ─────────────────────────────────────────────────────────
# Evaluate the best single-seed model through skore for the hub report.
best_single = HistGradientBoostingRegressor(
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=0,
    **best_params,
)
report = skore.evaluate(best_single, X_train_s2, y_train, splitter=cv_splits)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("09_tuned", report)

# %%  ── Fit on all train (5-seed ensemble), write submission ───────────────
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

test_preds_by_seed = []
for seed in range(N_SEEDS):
    m = HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        **best_params,
    )
    m.fit(X_train_s2, y_train)
    test_preds_by_seed.append(m.predict(X_test_s2))

final_preds = np.mean(np.stack(test_preds_by_seed, axis=0), axis=0)

submission = X_test_raw.reset_index()[["Index"]].copy()
submission["target"] = final_preds
submission.to_csv(submissions_dir / "09_tuned.csv", index=False)
print(f"\nSubmission written: {submissions_dir / '09_tuned.csv'} ({len(submission)} rows)")

# %% [markdown]
# # Experiment: 08b_stage2_push — push Variant A, write submission
#
# Reuses all setup from 08b_stage2.py (no gate, no re-evaluation of CV).
# Skips re-running CV (result already known: 3.9348 ± 0.0669).
# Pushes Variant A report to hub under key '08b_stage2'.
# Verifies train/test feature matrices share the same columns before fitting.

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features
from parkinson.hub import load_skore_credentials

# %%  ── Data & splits (identical to 07_trend_features.py / 08b_stage2.py) ───
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

X_train_feat = add_patient_features(X_train_raw)
X_test_feat  = add_patient_features(X_test_raw)

groups  = X_train_feat["patient_id"]
X_tr07  = X_train_feat.drop(columns=["patient_id"])
X_te07  = X_test_feat.drop(columns=["patient_id"])

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

# %%  ── Stage-1 predictions ──────────────────────────────────────────────────
s1_train = pd.read_csv(DATA_DIR / "stage1_train.csv", index_col="Index")["stage1"]
s1_test  = pd.read_csv(DATA_DIR / "stage1_test.csv",  index_col="Index")["stage1"]
s1_train = s1_train.reindex(X_train_raw.index)
s1_test  = s1_test.reindex(X_test_raw.index)

# %%  ── Stage-2 feature helpers ───────────────────────────────────────────────

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
        lin_slopes[pid] = sl
        lin_intercepts[pid] = ic
        if len(grp) >= 5:
            a, b, c = _quad_trend(grp["p"], grp["_age"])
        else:
            a, b, c = 0.0, sl, ic
        quad_a[pid], quad_b[pid], quad_c[pid] = a, b, c

    out["p_trend_slope"]  = out["_pid"].map(lin_slopes)
    out["p_trend_fitted"] = out["p_trend_slope"] * out["_age"] + out["_pid"].map(lin_intercepts)
    out["p_quad_fitted"]  = (
        out["_pid"].map(quad_a) * out["_age"] ** 2
        + out["_pid"].map(quad_b) * out["_age"]
        + out["_pid"].map(quad_c)
    )
    out = out.drop(columns=["_pid", "_age"])
    return out.sort_index()


# %%  ── Build stage-2 frames ──────────────────────────────────────────────────
_DROP_STR = ["cohort", "gene", "rater_id"]

X_train_s2 = add_stage1_features(X_tr07, groups, X_train_feat["age"], s1_train)
X_test_s2  = add_stage1_features(X_te07, X_test_feat["patient_id"], X_test_feat["age"], s1_test)

X_tr_s2 = X_train_s2.drop(columns=_DROP_STR, errors="ignore")
X_te_s2 = X_test_s2.drop(columns=_DROP_STR, errors="ignore")

# %%  ── Column-order check ────────────────────────────────────────────────────
assert list(X_tr_s2.columns) == list(X_te_s2.columns), (
    f"Column mismatch!\n"
    f"  train-only: {set(X_tr_s2.columns) - set(X_te_s2.columns)}\n"
    f"  test-only:  {set(X_te_s2.columns) - set(X_tr_s2.columns)}"
)
print(f"Column check OK — {X_tr_s2.shape[1]} features, same order in train and test.")

# %%  ── Push Variant A report to hub ──────────────────────────────────────────
hgbr   = HistGradientBoostingRegressor(random_state=0)
report = skore.evaluate(clone(hgbr), X_tr_s2, y_train, splitter=cv_splits)

cfg     = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("08b_stage2", report)

summary  = report.metrics.summarize().frame()
col_mean = [c for c in summary.columns if "mean" in c][0]
col_std  = [c for c in summary.columns if "std"  in c][0]
rmse_mean = float(summary.loc["rmse", col_mean])
rmse_std  = float(summary.loc["rmse", col_std])
print(f"08b_stage2 RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

# %%  ── Fit on all train, predict test, write submission ──────────────────────
final_model = clone(hgbr).fit(X_tr_s2, y_train)
test_preds  = final_model.predict(X_te_s2)

print(f"Test prediction mean: {test_preds.mean():.4f}  (expected ~37.5)")

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission = X_test_raw.reset_index()[["Index"]].copy()
submission["target"] = test_preds
assert len(submission) == 11013, f"Expected 11013 rows, got {len(submission)}"
submission.to_csv(submissions_dir / "08b_stage2.csv", index=False)
print(f"Submission written: {submissions_dir / '08b_stage2.csv'} ({len(submission)} rows)")

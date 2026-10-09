# %% [markdown]
# # Experiment: 08c_stage2_plus
#
# Stage 1: ensemble of 3 visit-level HGBR models (seeds 0/1/2, leaves 15/31/63,
#   lr=0.05); averaged OOF and test predictions → p.
# Stage 2: all 08b features + LOO Gaussian kernel smoothers (bw 0.5/1/2),
#   LOO mean/std/dev, LOO linear fit of p vs age.
# Model: tuned HGBR (lr=0.05, 600 iter, early stopping, leaves=15, min_samp=50, l2=1).

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import skrub

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

# ──────────────────────────────────────────────────────────────────────────────
# Data & splits  (identical to 07 / 08a / 08b)
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

# ──────────────────────────────────────────────────────────────────────────────
# Stage-1 feature set  (visit-level, same as 08a)
# ──────────────────────────────────────────────────────────────────────────────
S1_COLS = [
    "off", "on", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "age", "age_at_diagnosis",
    "sexM", "cohort", "gene", "rater_id",
]

def _make_s1_frame(X_raw):
    X = X_raw[S1_COLS].copy()
    X["on_minus_off"]     = X_raw["on"] - X_raw["off"]
    X["disease_duration"] = X_raw["age"] - X_raw["age_at_diagnosis"]
    return X

X_tr_s1 = _make_s1_frame(X_train_raw)
X_te_s1 = _make_s1_frame(X_test_raw)

# ──────────────────────────────────────────────────────────────────────────────
# Stage-1 ensemble: 3 models (seed×leaves), lr=0.05; average OOF & test preds
# ──────────────────────────────────────────────────────────────────────────────
S1_CONFIGS = [(0, 15), (1, 31), (2, 63)]  # (random_state, max_leaf_nodes)

tv = skrub.TableVectorizer(cardinality_threshold=100)

oof_stack  = np.zeros(len(y_train))
test_stack = np.zeros(len(X_test_raw))

for rs, leaves in S1_CONFIGS:
    hgbr_s1 = HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=leaves, random_state=rs
    )
    model_s1 = Pipeline([("tv", tv), ("hgbr", hgbr_s1)])

    oof = np.full(len(y_train), np.nan)
    for tr_idx, val_idx in cv_splits:
        m = clone(model_s1)
        m.fit(X_tr_s1.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = m.predict(X_tr_s1.iloc[val_idx])

    full = clone(model_s1).fit(X_tr_s1, y_train)
    oof_stack  += oof
    test_stack += full.predict(X_te_s1)

p_train = pd.Series(oof_stack  / len(S1_CONFIGS), index=X_train_raw.index)
p_test  = pd.Series(test_stack / len(S1_CONFIGS), index=X_test_raw.index)

oof_rmse = float(root_mean_squared_error(y_train, p_train))
print(f"Stage-1 ensemble OOF RMSE: {oof_rmse:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers
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


def _loo_features_group(ages: np.ndarray, vals: np.ndarray, bandwidths=(0.5, 1.0, 2.0)):
    """Return dict of leave-one-out feature arrays for one patient group.

    ages, vals: 1-D arrays aligned to the group rows (already sorted by age).
    Returns arrays of length len(ages).
    """
    n = len(ages)
    results = {f"p_loo_k{bw}": np.full(n, np.nan) for bw in bandwidths}
    results["p_loo_ksum"] = np.full(n, np.nan)
    results["p_loo_mean"] = np.full(n, np.nan)
    results["p_loo_std"]  = np.full(n, np.nan)
    results["p_loo_dev"]  = np.full(n, np.nan)
    results["p_loo_lin_fitted"] = np.full(n, np.nan)
    results["p_loo_lin_slope"]  = np.full(n, np.nan)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        others_age = ages[mask]
        others_val = vals[mask]
        valid = ~np.isnan(others_val)
        oa = others_age[valid]
        ov = others_val[valid]
        m = len(oa)

        # Gaussian kernel smoothers
        for bw in bandwidths:
            w = np.exp(-0.5 * ((ages[i] - oa) / bw) ** 2)
            ws = w.sum()
            results[f"p_loo_k{bw}"][i] = (w * ov).sum() / ws if ws > 0 else np.nan
        results["p_loo_ksum"][i] = np.exp(-0.5 * ((ages[i] - oa) / 1.0) ** 2).sum() if m > 0 else np.nan

        # LOO mean / std / dev
        if m > 0:
            mu = ov.mean()
            results["p_loo_mean"][i] = mu
            results["p_loo_std"][i]  = ov.std() if m > 1 else 0.0
            results["p_loo_dev"][i]  = (vals[i] if not np.isnan(vals[i]) else np.nan) - mu

        # LOO linear fit (fall back to LOO mean when < 3 other visits)
        if m >= 3:
            xc = oa - oa.mean()
            denom = (xc**2).sum()
            if denom > 0:
                slope = (xc * ov).sum() / denom
                intercept = ov.mean() - slope * oa.mean()
                results["p_loo_lin_fitted"][i] = slope * ages[i] + intercept
                results["p_loo_lin_slope"][i]  = slope
            else:
                results["p_loo_lin_fitted"][i] = ov.mean()
                results["p_loo_lin_slope"][i]  = 0.0
        elif m > 0:
            results["p_loo_lin_fitted"][i] = ov.mean()
            results["p_loo_lin_slope"][i]  = 0.0

    return results


def add_stage1_features(X07, patient_ids, ages, s1):
    """All 08b features + LOO features derived from p."""
    out = X07.copy()
    out["_pid"] = patient_ids.values
    out["_age"] = ages.values
    out["p"]    = s1.values

    out = out.sort_values(["_pid", "_age"])

    # ── 08b features ──────────────────────────────────────────────────────
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
        if len(grp) >= 5:
            a, b, c = _quad_trend(grp["p"], grp["_age"])
        else:
            a, b, c = 0.0, sl, ic
        quad_a[pid] = a; quad_b[pid] = b; quad_c[pid] = c

    out["p_trend_slope"]  = out["_pid"].map(lin_slopes)
    out["p_trend_fitted"] = out["p_trend_slope"] * out["_age"] + out["_pid"].map(lin_intercepts)
    out["p_quad_fitted"]  = (
        out["_pid"].map(quad_a) * out["_age"]**2
        + out["_pid"].map(quad_b) * out["_age"]
        + out["_pid"].map(quad_c)
    )

    # ── new LOO features ──────────────────────────────────────────────────
    loo_cols = [
        "p_loo_k0.5", "p_loo_k1.0", "p_loo_k2.0", "p_loo_ksum",
        "p_loo_mean", "p_loo_std", "p_loo_dev",
        "p_loo_lin_fitted", "p_loo_lin_slope",
    ]
    loo_buf = {c: np.full(len(out), np.nan) for c in loo_cols}

    # Work on the position-indexed (sorted) frame
    pos = np.arange(len(out))
    pid_arr = out["_pid"].values
    age_arr = out["_age"].values
    p_arr   = out["p"].values

    unique_pids, pid_inv = np.unique(pid_arr, return_inverse=True)
    for uid_idx, pid in enumerate(unique_pids):
        rows = np.where(pid_inv == uid_idx)[0]
        grp_ages = age_arr[rows]
        grp_vals = p_arr[rows]
        res = _loo_features_group(grp_ages, grp_vals)
        for col in loo_cols:
            loo_buf[col][rows] = res[col]

    for col in loo_cols:
        out[col] = loo_buf[col]

    out = out.drop(columns=["_pid", "_age"])
    return out.sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# Build stage-2 frames
# ──────────────────────────────────────────────────────────────────────────────
_DROP_STR = ["cohort", "gene", "rater_id"]

X_tr_s2 = add_stage1_features(X_tr07, groups, X_train_feat["age"], p_train).drop(columns=_DROP_STR, errors="ignore")
X_te_s2 = add_stage1_features(X_te07, X_test_feat["patient_id"], X_test_feat["age"], p_test).drop(columns=_DROP_STR, errors="ignore")

assert list(X_tr_s2.columns) == list(X_te_s2.columns), "Column mismatch train/test"

# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 model
# ──────────────────────────────────────────────────────────────────────────────
hgbr_s2 = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=600,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    max_leaf_nodes=15,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=0,
)

fold_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(hgbr_s2)
    m.fit(X_tr_s2.iloc[tr_idx], y_train.iloc[tr_idx])
    fold_rmses.append(float(root_mean_squared_error(
        y_train.iloc[val_idx], m.predict(X_tr_s2.iloc[val_idx])
    )))

rmse_mean = float(np.mean(fold_rmses))
rmse_std  = float(np.std(fold_rmses))

print(f"08b Variant A          RMSE: 3.9348 ± 0.0669")
print(f"08c_stage2_plus        RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")

if rmse_mean >= 3.88:
    print(f"RMSE {rmse_mean:.4f} ≥ 3.88 — stopping.")
else:
    print(f"RMSE {rmse_mean:.4f} < 3.88 — pushing to hub.")

    import skore
    from skore import Project, login
    from parkinson.hub import load_skore_credentials

    report = skore.evaluate(clone(hgbr_s2), X_tr_s2, y_train, splitter=cv_splits)

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("08c_stage2_plus", report)
    print(f"Report URL: https://skore.probabl.ai/c2f/bobathon-esilv")

    # Fit on all train averaging 3 seeds
    test_preds = np.zeros(len(X_test_raw))
    for seed in range(3):
        m = clone(hgbr_s2)
        m.set_params(random_state=seed)
        m.fit(X_tr_s2, y_train)
        test_preds += m.predict(X_te_s2)
    test_preds /= 3

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission = X_test_raw.reset_index()[["Index"]].copy()
    submission["target"] = test_preds
    submission.to_csv(submissions_dir / "08c_stage2_plus.csv", index=False)
    print(f"Submission written: {submissions_dir / '08c_stage2_plus.csv'} ({len(submission)} rows)")

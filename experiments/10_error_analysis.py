# %% [markdown]
# # Experiment: 10_error_analysis — OOF error analysis of 08b Variant A

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

# ──────────────────────────────────────────────────────────────────────────────
# Data & splits  (verbatim from 08b_stage2.py)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

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
    out = out.drop(columns=["_pid", "_age"])
    return out.sort_index()


X_train_s2 = add_stage1_features(X_tr07, groups, X_train_feat["age"], s1_train)
_DROP_STR   = ["cohort", "gene", "rater_id"]
X_tr_s2     = X_train_s2.drop(columns=_DROP_STR, errors="ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Recompute Variant A OOF predictions
# ──────────────────────────────────────────────────────────────────────────────
hgbr    = HistGradientBoostingRegressor(random_state=0)
oof_pred = np.full(len(y_train), np.nan)

for tr_idx, val_idx in cv_splits:
    m = clone(hgbr)
    m.fit(X_tr_s2.iloc[tr_idx], y_train.iloc[tr_idx])
    oof_pred[val_idx] = m.predict(X_tr_s2.iloc[val_idx])

# ──────────────────────────────────────────────────────────────────────────────
# Build diagnostic frame
# ──────────────────────────────────────────────────────────────────────────────
d = X_train_raw[["patient_id", "rater_id", "cohort", "off", "on",
                 "age", "age_at_diagnosis"]].copy()
d["target"]          = y_train.values
d["pred"]            = oof_pred
d["error"]           = d["pred"] - d["target"]
d["sq_error"]        = d["error"] ** 2
d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
d["n_visits"]        = d.groupby("patient_id")["target"].transform("count")

# Save OOF file
oof_out = d[["patient_id", "rater_id", "target", "pred"]].copy()
oof_out.index.name = "Index"
oof_out.to_csv(DATA_DIR / "stage2_oof.csv")

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
SEP = "-" * 62

def table(df, label_col, label_order=None):
    rows = []
    groups_iter = df.groupby(label_col, observed=True)
    for lbl, g in groups_iter:
        r = float(root_mean_squared_error(g["target"], g["pred"]))
        rows.append((lbl, len(g), r))
    if label_order:
        rows = sorted(rows, key=lambda x: label_order.index(x[0]) if x[0] in label_order else 99)
    hdr = f"  {'Group':<24}  {'n':>6}  {'RMSE':>7}"
    print(hdr)
    print("  " + "-"*42)
    for lbl, n, r in rows:
        print(f"  {str(lbl):<24}  {n:>6}  {r:>7.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Slice tables
# ──────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Visits per patient")
print(SEP)
bins   = [0, 3, 6, 9, 9999]
labels = ["1-3", "4-6", "7-9", "10+"]
d["visit_bucket"] = pd.cut(d["n_visits"], bins=bins, labels=labels)
table(d, "visit_bucket", labels)

print(SEP)
print("off measured vs missing")
print(SEP)
d["off_status"] = np.where(d["off"].notna(), "measured", "missing")
table(d, "off_status", ["measured", "missing"])

print(SEP)
print("on measured vs missing")
print(SEP)
d["on_status"] = np.where(d["on"].notna(), "measured", "missing")
table(d, "on_status", ["measured", "missing"])

print(SEP)
print("Cohort")
print(SEP)
table(d, "cohort")

print(SEP)
print("disease_duration quartiles")
print(SEP)
d["dd_q"] = pd.qcut(d["disease_duration"], 4,
                    labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
table(d, "dd_q", ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])

print(SEP)
print("target quartiles")
print(SEP)
d["tgt_q"] = pd.qcut(d["target"], 4,
                     labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
table(d, "tgt_q", ["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

# ──────────────────────────────────────────────────────────────────────────────
# Mean signed error per target quartile
# ──────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Mean signed error (pred - target) by target quartile")
print(SEP)
print(f"  {'Quartile':<20}  {'mean error':>10}  {'std error':>10}")
print("  " + "-"*44)
for lbl, g in d.groupby("tgt_q", observed=True):
    print(f"  {str(lbl):<20}  {g['error'].mean():>+10.4f}  {g['error'].std():>10.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Std of rater-level mean signed error
# ──────────────────────────────────────────────────────────────────────────────
rater_mean_error = d.groupby("rater_id")["error"].mean()
print(SEP)
print(f"Std across rater_id of mean(pred - target): {rater_mean_error.std():.4f}")
print(f"  (n raters={len(rater_mean_error)},  "
      f"min={rater_mean_error.min():+.4f},  "
      f"max={rater_mean_error.max():+.4f})")
print(SEP)

# ──────────────────────────────────────────────────────────────────────────────
# 10 worst patients (largest mean squared error)
# ──────────────────────────────────────────────────────────────────────────────
pat = (
    d.groupby("patient_id")
    .agg(
        mse       =("sq_error", "mean"),
        n_visits  =("n_visits", "first"),
        cohort    =("cohort",   "first"),
        off_frac  =("off",      lambda x: x.notna().mean()),
    )
    .sort_values("mse", ascending=False)
    .head(10)
)
pat["rmse"] = np.sqrt(pat["mse"])

print("Top 10 patients by mean squared error")
print(SEP)
print(f"  {'patient_id':<14}  {'n':>4}  {'cohort':>6}  {'off_frac':>8}  {'RMSE':>7}")
print("  " + "-"*48)
for pid, row in pat.iterrows():
    print(f"  {pid:<14}  {int(row['n_visits']):>4}  {row['cohort']:>6}  "
          f"{row['off_frac']:>8.2f}  {row['rmse']:>7.4f}")
print(SEP)

# %% [markdown]
# # Experiment: 20_the_secret_hack
#
# Within-patient trajectory approach:
#
# Key insight confirmed by diag_07:
#   - Median per-patient LOO RMSE of a linear fit (age → target) = 0.94
#   - But at TEST time we have no target — we must use features (off, imputed_off)
#     as a proxy for the trajectory.
#
# Strategy:
#   1. Train a cross-patient scaler:  target ~ off_trendline  (OLS, per patient)
#      i.e. learn (patient-level trendline slope & intercept of off vs age) →
#      (patient-level trendline slope & intercept of target vs age).
#      This is a 4-feature linear model: [s_off, i_off, s_off², i_off²]
#
#   2. For test patients: compute their per-patient linear fit of imputed_off
#      vs age → get (s_off_test, i_off_test) → predict (s_target, i_target)
#      using the model from step 1.
#
#   3. Final prediction at age a: s_target * a + i_target + rater_bias
#
# Evaluation:  Train LOO (per-patient linear fit of TARGET vs age, leaving one
#              visit out) gives the ceiling; the proxy approach gives the
#              realistic test estimate.

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

# ── Imputed off (same formula as exp 13) ─────────────────────────────────────
def add_imputed_off(df: pd.DataFrame) -> np.ndarray:
    diff = df["off"] - df["on"]
    pat_mean_diff = pd.Series(diff.values, index=df.index).groupby(
        df["patient_id"].values).transform("mean")
    imp = np.where(df["off"].notna(), df["off"], df["on"] + pat_mean_diff.values)
    pat_mean_imp = pd.Series(imp, index=df.index).groupby(
        df["patient_id"].values).transform("mean")
    imp = np.where(np.isnan(imp), pat_mean_imp.values, imp)
    return imp

X_train["imputed_off"] = add_imputed_off(X_train)
X_test["imputed_off"]  = add_imputed_off(X_test)
X_train["target"]      = y_train


# ── OLS helper ───────────────────────────────────────────────────────────────
def fit_line(ages, vals):
    """Return (slope, intercept) via OLS, or (0, nanmean) with < 2 valid pts."""
    mask = ~np.isnan(vals) & ~np.isnan(ages)
    x, y = ages[mask], vals[mask]
    if len(x) < 2:
        return 0.0, float(np.nanmean(vals)) if len(vals) > 0 else 0.0
    xc = x - x.mean()
    d  = (xc**2).sum()
    if d == 0:
        return 0.0, float(y.mean())
    s = float((xc * y).sum() / d)
    return s, float(y.mean() - s * x.mean())


# ── Step A: ceiling — LOO linear fit of TARGET vs age on train ───────────────
loo_target = np.full(len(X_train), np.nan)
for pid, grp in X_train.groupby("patient_id"):
    ages   = grp["age"].values.astype(float)
    target = grp["target"].values.astype(float)
    n      = len(grp)
    pos    = np.where(X_train.index.isin(grp.index))[0]
    if n == 1:
        loo_target[pos[0]] = target[0]
        continue
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        s, ic     = fit_line(ages[mask], target[mask])
        loo_target[pos[i]] = s * ages[i] + ic

ceiling_rmse = float(root_mean_squared_error(y_train, loo_target))
print(f"Ceiling — LOO linear fit (age → target) RMSE: {ceiling_rmse:.4f}")


# ── Step B: rater bias on train ──────────────────────────────────────────────
# bias[rater] = mean(target - imputed_off) per rater.
rater_diff   = X_train["target"] - X_train["imputed_off"]
rater_bias   = rater_diff.groupby(X_train["rater_id"]).mean().to_dict()
global_bias  = float(rater_diff.mean())


# ── Step C: per-patient line params (train) ───────────────────────────────────
# For each train patient: fit (slope, intercept) of imputed_off vs age
#                      and (slope, intercept) of target vs age.
# These will train a meta-model: off_params → target_params.
pat_off_params    = {}  # pid → (s_off, i_off)
pat_target_params = {}  # pid → (s_tgt, i_tgt)

for pid, grp in X_train.groupby("patient_id"):
    ages = grp["age"].values.astype(float)
    pat_off_params[pid]    = fit_line(ages, grp["imputed_off"].values.astype(float))
    pat_target_params[pid] = fit_line(ages, grp["target"].values.astype(float))

# ── Step D: meta-model (off_params → target_params) ──────────────────────────
pids    = list(pat_off_params.keys())
_global_off_mean = float(X_train["imputed_off"].mean())

def _safe_feat(s, i):
    s = 0.0 if np.isnan(s) else s
    i = _global_off_mean if np.isnan(i) else i
    return [s, i, s**2, i**2]

X_meta  = np.array([_safe_feat(*pat_off_params[p]) for p in pids])
y_meta_slope = np.array([pat_target_params[p][0] for p in pids])
y_meta_intcp = np.array([pat_target_params[p][1] for p in pids])

meta_slope = Ridge(alpha=1.0).fit(X_meta, y_meta_slope)
meta_intcp = Ridge(alpha=1.0).fit(X_meta, y_meta_intcp)

print(f"Meta-model R² (slope): {meta_slope.score(X_meta, y_meta_slope):.4f}")
print(f"Meta-model R² (intcp): {meta_intcp.score(X_meta, y_meta_intcp):.4f}")


# ── Step E: proxy predictions on train (using off_params → predicted tgt line) ─
proxy_train = np.full(len(X_train), np.nan)
for pid, grp in X_train.groupby("patient_id"):
    pos  = np.where(X_train.index.isin(grp.index))[0]
    ages = grp["age"].values.astype(float)
    s_off, i_off = pat_off_params[pid]
    feat = np.array([_safe_feat(s_off, i_off)])
    s_tgt = float(meta_slope.predict(feat)[0])
    i_tgt = float(meta_intcp.predict(feat)[0])
    rates = grp["rater_id"].values
    for j, i in enumerate(pos):
        rbias = rater_bias.get(rates[j], global_bias)
        proxy_train[i] = s_tgt * ages[j] + i_tgt + (rbias - global_bias)

proxy_rmse = float(root_mean_squared_error(y_train, proxy_train))
print(f"\nProxy train RMSE (off→tgt meta-model + rater bias): {proxy_rmse:.4f}")
print(f"(Compare: ceiling LOO={ceiling_rmse:.4f}, best HGBR=3.7453)")


# ── Step F: test predictions ─────────────────────────────────────────────────
test_pred = np.full(len(X_test), np.nan)
for pid, grp in X_test.groupby("patient_id"):
    pos  = np.where(X_test.index.isin(grp.index))[0]
    ages = grp["age"].values.astype(float)
    s_off, i_off = fit_line(ages, grp["imputed_off"].values.astype(float))
    feat  = np.array([_safe_feat(s_off, i_off)])
    s_tgt = float(meta_slope.predict(feat)[0])
    i_tgt = float(meta_intcp.predict(feat)[0])
    rates = grp["rater_id"].values
    for j, i in enumerate(pos):
        rbias = rater_bias.get(rates[j], global_bias)
        test_pred[i] = s_tgt * ages[j] + i_tgt + (rbias - global_bias)

# Fill any remaining NaN with global mean
test_pred = np.where(np.isnan(test_pred), float(y_train.mean()), test_pred)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission = pd.DataFrame({"Index": X_test.index, "target": test_pred})
submission.to_csv(submissions_dir / "20_the_secret_hack.csv", index=False)

print(f"\nTest prediction mean  : {test_pred.mean():.4f}  (train target mean: {y_train.mean():.4f})")
print(f"Test prediction std   : {test_pred.std():.4f}   (train target std:  {y_train.std():.4f})")
print(f"Submission written    : {submissions_dir / '20_the_secret_hack.csv'} ({len(submission)} rows)")

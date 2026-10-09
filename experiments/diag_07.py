# %% [markdown]
# # Diagnostics: diag_07 — OOF analysis of 07_trend_features
#
# **Date:** 2025-07-10
# **Purpose:**
#   Part 1 — Slice the out-of-fold residuals from 07_trend_features by
#     measurement status, cohort, visit-count bucket, and rater_id to surface
#     where the model is weakest.
#   Part 2 — Check the encoder TableVectorizer chose for rater_id and gene,
#     then compare a version with cardinality_threshold=100 (forces rater_id
#     to low-cardinality / OneHot) against the default (StringEncoder).

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import skrub

import skore

from parkinson import PROJECT_ROOT
from parkinson.features import add_patient_features

# %%  ── Data (same preprocessing as experiment 07) ──────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

X_train_feat = add_patient_features(X_train_raw)
groups = X_train_feat["patient_id"]

DROP_COLS = ["patient_id"]
X_train = X_train_feat.drop(columns=DROP_COLS)

cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# %%  ── Build OOF predictions ─────────────────────────────────────────────
# Run GroupKFold CV manually so we have the full OOF prediction array.

model = skrub.tabular_pipeline("regressor")

oof_pred = np.full(len(y_train), np.nan)
for train_idx, val_idx in cv_splits:
    m = clone(model)
    m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    oof_pred[val_idx] = m.predict(X_train.iloc[val_idx])

# Build a diagnostic frame that keeps raw columns (off, on, cohort, rater_id)
# aligned with OOF predictions and residuals.
diag = X_train_raw.copy()
diag["y_true"] = y_train
diag["y_pred"] = oof_pred
diag["residual"] = diag["y_true"] - diag["y_pred"]
diag["n_visits"] = diag.groupby("patient_id")["y_true"].transform("count")

# Helper: compute RMSE from two aligned series.
def rmse(true, pred):
    return float(root_mean_squared_error(true, pred))


# =========================================================================
# Part 1a — RMSE by measurement-status and cohort/visit-count buckets
# =========================================================================

print("=" * 70)
print("PART 1a — RMSE by subgroup")
print("=" * 70)

# --- off measured vs missing ------------------------------------------------
for label, mask in [
    ("off measured    ", diag["off"].notna()),
    ("off missing     ", diag["off"].isna()),
    ("on  measured    ", diag["on"].notna()),
    ("on  missing     ", diag["on"].isna()),
]:
    sub = diag[mask]
    r = rmse(sub["y_true"], sub["y_pred"]) if len(sub) >= 2 else float("nan")
    print(f"  {label}  n={len(sub):6d}  RMSE={r:.4f}")

print()

# --- per cohort -------------------------------------------------------------
print("  Cohort RMSE:")
for cohort, grp in diag.groupby("cohort"):
    r = rmse(grp["y_true"], grp["y_pred"])
    print(f"    {cohort:15s}  n={len(grp):6d}  RMSE={r:.4f}")

print()

# --- by number-of-visits bucket --------------------------------------------
print("  Visits-bucket RMSE:")
bins = [0, 2, 4, 8, 16, 9999]
labels_b = ["1–2", "3–4", "5–8", "9–16", "17+"]
diag["visits_bucket"] = pd.cut(diag["n_visits"], bins=bins, labels=labels_b)
for bucket, grp in diag.groupby("visits_bucket", observed=True):
    r = rmse(grp["y_true"], grp["y_pred"])
    print(f"    {str(bucket):8s}  n={len(grp):6d}  RMSE={r:.4f}")

print()

# =========================================================================
# Part 1b — Bias (mean and std of off − target) overall, per cohort,
#           and top-5 / bottom-5 rater_id means
# =========================================================================

print("=" * 70)
print("PART 1b — Bias: mean and std of (off − target)")
print("=" * 70)

diag["off_minus_target"] = diag["off"] - diag["y_true"]

overall = diag["off_minus_target"].dropna()
print(f"\n  Overall       mean={overall.mean():.4f}  std={overall.std():.4f}"
      f"  (n non-null={len(overall)})")

print("\n  Per cohort:")
for cohort, grp in diag.groupby("cohort"):
    s = grp["off_minus_target"].dropna()
    if len(s) == 0:
        continue
    print(f"    {cohort:15s}  mean={s.mean():+.4f}  std={s.std():.4f}  (n={len(s)})")

print("\n  Per rater_id (top-5 highest mean, top-5 lowest mean):")
rater_stats = (
    diag.groupby("rater_id")["off_minus_target"]
    .agg(mean="mean", std="std", count="count")
    .dropna(subset=["mean"])
    .sort_values("mean")
)
top5 = rater_stats.tail(5).iloc[::-1]
bot5 = rater_stats.head(5)
print("    [Highest 5 rater means]")
for rid, row in top5.iterrows():
    print(f"      {rid:10s}  mean={row['mean']:+.4f}  std={row['std']:.4f}  n={int(row['count'])}")
print("    [Lowest 5 rater means]")
for rid, row in bot5.iterrows():
    print(f"      {rid:10s}  mean={row['mean']:+.4f}  std={row['std']:.4f}  n={int(row['count'])}")

print()

# =========================================================================
# Part 1c — RMSE of a per-patient linear fit of target vs age
#           (patients with ≥ 4 visits)
# =========================================================================

print("=" * 70)
print("PART 1c — RMSE of per-patient linear fit: target ~ age (≥4 visits)")
print("=" * 70)

def patient_linear_rmse(grp):
    """OLS of target on age within the group; returns RMSE of fitted values."""
    ages = grp["age"].values.astype(float)
    y = grp["y_true"].values.astype(float)
    if len(y) < 2 or np.std(ages) == 0:
        return np.nan
    x_c = ages - ages.mean()
    slope = (x_c * y).sum() / (x_c ** 2).sum()
    fitted = ages * slope + (y.mean() - slope * ages.mean())
    return float(root_mean_squared_error(y, fitted))

eligible = diag.groupby("patient_id").filter(lambda g: len(g) >= 4)
per_patient_rmse = eligible.groupby("patient_id").apply(
    patient_linear_rmse, include_groups=False
)
# include_groups not available in older pandas; fall back if needed
if isinstance(per_patient_rmse, pd.DataFrame):
    per_patient_rmse = eligible.groupby("patient_id")[["age", "y_true"]].apply(
        lambda g: patient_linear_rmse(g.rename(columns={"y_true": "y_true", "age": "age"}))
    )

valid = per_patient_rmse.dropna()
print(
    f"\n  Patients with ≥4 visits: {len(valid)}"
    f"  Median per-patient linear RMSE: {valid.median():.4f}"
    f"  Mean: {valid.mean():.4f}"
    f"  Std: {valid.std():.4f}"
)
print()

# =========================================================================
# Part 2 — TableVectorizer encoder choices for rater_id and gene,
#          then compare default (StringEncoder for rater_id) vs
#          cardinality_threshold=100 (OneHot for rater_id)
# =========================================================================

print("=" * 70)
print("PART 2 — TableVectorizer encoder inspection + cardinality comparison")
print("=" * 70)

# --- Fit one fold to get an inspectable TableVectorizer --------------------
m_inspect = clone(model)
tr_idx, _ = cv_splits[0]
m_inspect.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])

tv = m_inspect.named_steps["tablevectorizer"]
rater_kind = tv.column_to_kind_.get("rater_id", "n/a")
gene_kind   = tv.column_to_kind_.get("gene", "n/a")
rater_encoder = type(tv.transformers_.get("rater_id", None)).__name__
gene_encoder  = type(tv.transformers_.get("gene", None)).__name__
rater_card = int(X_train_raw["rater_id"].nunique())
gene_card  = int(X_train_raw["gene"].nunique())

print(
    f"\n  rater_id  cardinality={rater_card:4d}  kind={rater_kind:20s}  encoder={rater_encoder}"
)
print(
    f"  gene      cardinality={gene_card:4d}  kind={gene_kind:20s}  encoder={gene_encoder}"
)
print()

# --- Default model (threshold=40, rater_id → StringEncoder) ---------------
default_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(model)
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_train.iloc[val_idx])
    default_rmses.append(rmse(y_train.iloc[val_idx], preds))

default_mean = float(np.mean(default_rmses))
default_std  = float(np.std(default_rmses))

# --- Low-cardinality model (threshold=100, rater_id → OneHot) --------------
tv_lowcard = skrub.TableVectorizer(cardinality_threshold=100)
model_lowcard = Pipeline([
    ("tablevectorizer", tv_lowcard),
    ("histgradientboostingregressor", HistGradientBoostingRegressor()),
])

# Verify the threshold flips rater_id to low-cardinality
m_check = clone(model_lowcard)
m_check.fit(X_train.iloc[cv_splits[0][0]], y_train.iloc[cv_splits[0][0]])
tv_check = m_check.named_steps["tablevectorizer"]
rater_kind_100 = tv_check.column_to_kind_.get("rater_id", "n/a")
rater_enc_100  = type(tv_check.transformers_.get("rater_id", None)).__name__
print(f"  With threshold=100: rater_id kind={rater_kind_100}  encoder={rater_enc_100}")
print()

lowcard_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(model_lowcard)
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_train.iloc[val_idx])
    lowcard_rmses.append(rmse(y_train.iloc[val_idx], preds))

lowcard_mean = float(np.mean(lowcard_rmses))
lowcard_std  = float(np.std(lowcard_rmses))

print(f"  Default  (threshold= 40, rater_id=StringEncoder)  RMSE: "
      f"{default_mean:.4f} ± {default_std:.4f}")
print(f"  LowCard  (threshold=100, rater_id=OneHot       )  RMSE: "
      f"{lowcard_mean:.4f} ± {lowcard_std:.4f}")

delta = lowcard_mean - default_mean
sign = "+" if delta >= 0 else ""
print(f"  Δ (LowCard − Default): {sign}{delta:.4f}")
print()

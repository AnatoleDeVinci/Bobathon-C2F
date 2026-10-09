# %% [markdown]
# # Experiment: 08_stack — 2-stage stacking on top of 07_trend_features
#
# **Date:** 2025-07-10
# **Stage 1:** tabular_pipeline (StringEncoder for rater_id, threshold=40 —
#   the better encoder from diag_07) on the 07 feature set.
#   Out-of-fold predictions on train; predictions on test from a model fit on
#   all train.  OOF RMSE is expected to be ~4.30.
#
# **Stage 2 features:** the 07 features PLUS per-visit and per-patient
#   meta-features derived from the stage-1 prediction:
#     • s1_pred:      stage-1 prediction at this visit
#     • prev1/2 and next1/2 of s1_pred (sorted by age within patient)
#     • s1_mean, s1_median, s1_std per patient
#     • s1_trend_slope and s1_trend_fitted (OLS of s1_pred vs age per patient)
#
# **Variant A:** stage-2 HGBR predicts the target directly.
# **Variant B:** stage-2 HGBR predicts the residual
#   (target − per-patient linear trend of stage-1 vs age), then adds the trend
#   back to recover the final prediction.
#
# Best variant is pushed to the hub as 08_stack and a submission is written.

# %%
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

# %%  ── Helpers ──────────────────────────────────────────────────────────────

def rmse(true, pred):
    return float(root_mean_squared_error(true, pred))


def add_stage1_meta(df: pd.DataFrame, s1_col: str = "s1_pred") -> pd.DataFrame:
    """Append per-patient meta-features derived from the stage-1 prediction.

    Parameters
    ----------
    df :
        Frame already sorted by (patient_id, age) with a column *s1_col*.
    s1_col :
        Name of the stage-1 prediction column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with stage-1 meta columns appended.
    """
    out = df.copy()

    # Lag / lead stage-1 predictions (visits already sorted by age)
    for k in (1, 2):
        out[f"s1_prev{k}"] = out.groupby("patient_id")[s1_col].shift(k)
        out[f"s1_next{k}"] = out.groupby("patient_id")[s1_col].shift(-k)

    # Per-patient statistics
    grp = out.groupby("patient_id")[s1_col]
    out["s1_mean"]   = grp.transform("mean")
    out["s1_median"] = grp.transform("median")
    out["s1_std"]    = grp.transform("std")

    # Per-patient linear trend of s1_pred vs age
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


# %%  ── Data ─────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

# 07 feature set computed independently on each frame
X_train_07 = add_patient_features(X_train_raw)
X_test_07  = add_patient_features(X_test_raw)

groups = X_train_07["patient_id"]

DROP_COLS = ["patient_id"]
X_tr07 = X_train_07.drop(columns=DROP_COLS)
X_te07 = X_test_07.drop(columns=DROP_COLS)

cv_splits = list(GroupKFold(n_splits=5).split(X_tr07, y_train, groups=groups))

# %%  ── Stage 1: tabular_pipeline (default threshold=40, StringEncoder) ─────
stage1_model = skrub.tabular_pipeline("regressor")

# OOF predictions on train
s1_oof = np.full(len(y_train), np.nan)
for tr_idx, val_idx in cv_splits:
    m = clone(stage1_model)
    m.fit(X_tr07.iloc[tr_idx], y_train.iloc[tr_idx])
    s1_oof[val_idx] = m.predict(X_tr07.iloc[val_idx])

s1_oof_rmse = rmse(y_train, s1_oof)
print(f"Stage-1 OOF RMSE: {s1_oof_rmse:.4f}  (expected ~4.30)")

# Stage-1 predictions on test (fit on all train)
s1_full = clone(stage1_model).fit(X_tr07, y_train)
s1_test_preds = s1_full.predict(X_te07)

# %%  ── Stage 2 feature construction ────────────────────────────────────────
# Sort by (patient_id, age) before computing lag/lead meta-features.

# String columns in the 07 feature set that HGBR cannot consume directly.
# Their signal is already baked into the stage-1 prediction, so we drop
# them from stage-2 features.
_S2_DROP_STRING = ["cohort", "gene", "rater_id"]


def build_stage2_frame(
    X07: pd.DataFrame,
    patient_ids: pd.Series,
    ages: pd.Series,
    s1_predictions: np.ndarray,
) -> pd.DataFrame:
    """Combine 07 numeric features with stage-1 meta-features.

    String columns (cohort, gene, rater_id) are dropped — their signal is
    already captured in s1_pred.  Returns a frame with the same positional
    index as *X07* (suitable for iloc-based CV splits).
    """
    frame = X07.drop(columns=_S2_DROP_STRING, errors="ignore").copy()
    frame["patient_id"] = patient_ids.values
    frame["age_raw"] = ages.values           # age already in X07; duplicate for sorting
    frame["s1_pred"] = s1_predictions

    # Sort by patient_id, age so shifts are chronological
    frame = frame.sort_values(["patient_id", "age_raw"])
    frame = add_stage1_meta(frame, s1_col="s1_pred")
    frame = frame.drop(columns=["patient_id", "age_raw"])
    # Restore original positional order so iloc CV splits stay aligned
    frame = frame.sort_index()
    return frame


X_train_s2 = build_stage2_frame(
    X_tr07, groups, X_train_07["age"], s1_oof
)
X_test_s2 = build_stage2_frame(
    X_te07,
    X_test_07["patient_id"],
    X_test_07["age"],
    s1_test_preds,
)

# %%  ── Variant A: stage-2 predicts target directly ────────────────────────
stage2_model = HistGradientBoostingRegressor()

varA_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(stage2_model)
    m.fit(X_train_s2.iloc[tr_idx], y_train.iloc[tr_idx])
    varA_rmses.append(rmse(y_train.iloc[val_idx], m.predict(X_train_s2.iloc[val_idx])))

varA_mean, varA_std = float(np.mean(varA_rmses)), float(np.std(varA_rmses))
print(f"Variant A (direct)   RMSE: {varA_mean:.4f} ± {varA_std:.4f}")

# %%  ── Variant B: stage-2 predicts residual ─────────────────────────────
# Residual = target − per-patient linear trend of stage-1 predictions.
# The per-patient trend is computed from the OOF s1_pred (train) and
# the full-train-fit s1_pred (test) — both already in X_train_s2 / X_test_s2
# as s1_trend_fitted.

residual_train = y_train - pd.Series(
    X_train_s2["s1_trend_fitted"].values, index=y_train.index
)

varB_rmses = []
for tr_idx, val_idx in cv_splits:
    m = clone(stage2_model)
    m.fit(X_train_s2.iloc[tr_idx], residual_train.iloc[tr_idx])
    resid_pred = m.predict(X_train_s2.iloc[val_idx])
    final_pred = resid_pred + X_train_s2["s1_trend_fitted"].iloc[val_idx].values
    varB_rmses.append(rmse(y_train.iloc[val_idx], final_pred))

varB_mean, varB_std = float(np.mean(varB_rmses)), float(np.std(varB_rmses))
print(f"Variant B (residual) RMSE: {varB_mean:.4f} ± {varB_std:.4f}")

# %%  ── Summary comparison ────────────────────────────────────────────────
print()
print(f"07_trend_features    RMSE: 4.3009 ± 0.0682  (from experiment 07 run)")
print(f"08 Variant A direct  RMSE: {varA_mean:.4f} ± {varA_std:.4f}")
print(f"08 Variant B residual RMSE: {varB_mean:.4f} ± {varB_std:.4f}")

best_label   = "A" if varA_mean <= varB_mean else "B"
best_mean    = varA_mean if varA_mean <= varB_mean else varB_mean
best_std     = varA_std  if varA_mean <= varB_mean else varB_std
best_model   = clone(stage2_model)
best_target  = y_train if best_label == "A" else residual_train
print(f"\nBest variant: {best_label}  RMSE {best_mean:.4f} ± {best_std:.4f}")

# %%  ── Push best to hub via skore.evaluate ─────────────────────────────────
# Re-run the best variant through skore.evaluate for the hub report.
best_cv_splits_data = cv_splits

if best_label == "A":
    report = skore.evaluate(
        clone(stage2_model), X_train_s2, y_train, splitter=best_cv_splits_data
    )
else:
    # For variant B we push the residual model; note in the report name.
    report = skore.evaluate(
        clone(stage2_model), X_train_s2, residual_train, splitter=best_cv_splits_data
    )

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("08_stack", report)

# %%  ── Fit stage 2 on all train, write submission ──────────────────────────
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

best_model.fit(X_train_s2, best_target)

if best_label == "A":
    test_final = best_model.predict(X_test_s2)
else:
    resid_test = best_model.predict(X_test_s2)
    test_final = resid_test + X_test_s2["s1_trend_fitted"].values

submission = X_test_raw.reset_index()[["Index"]].copy()
submission["target"] = test_final
submission.to_csv(submissions_dir / "08_stack.csv", index=False)
print(f"\nSubmission written: {submissions_dir / '08_stack.csv'} ({len(submission)} rows)")

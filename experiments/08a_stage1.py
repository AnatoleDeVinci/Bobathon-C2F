# %% [markdown]
# # Experiment: 08a_stage1 — visit-level debiasing model (no patient aggregates)
#
# Stage 1 of a 2-stage stack. Sees one visit at a time — no neighbour features,
# no per-patient aggregates. Produces OOF predictions on train and full-train
# predictions on test, saved to data/stage1_train.csv and data/stage1_test.csv.

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
from parkinson.features import add_patient_features

# %%  ── Data & splits (identical to experiments/07_trend_features.py) ───────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

# groups derived the same way as 07 (patient_id column of enriched frame)
X_train_feat = add_patient_features(X_train_raw)
groups = X_train_feat["patient_id"]

# Stage-1 feature columns — visit-level only, no aggregates
S1_COLS = [
    "off", "on", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "age", "age_at_diagnosis",
    "sexM", "cohort", "gene", "rater_id",
]

# on_minus_off is stateless and visit-level — compute it from raw data
X_train_s1 = X_train_raw[S1_COLS].copy()
X_train_s1["on_minus_off"]     = X_train_raw["on"] - X_train_raw["off"]
X_train_s1["disease_duration"] = X_train_raw["age"] - X_train_raw["age_at_diagnosis"]

X_test_s1 = X_test_raw[S1_COLS].copy()
X_test_s1["on_minus_off"]     = X_test_raw["on"] - X_test_raw["off"]
X_test_s1["disease_duration"] = X_test_raw["age"] - X_test_raw["age_at_diagnosis"]

# Same GroupKFold splits as experiments 03–07
# (split is computed on the enriched X_train to keep identical row alignment)
X_train_drop = X_train_feat.drop(columns=["patient_id"])
cv_splits = list(GroupKFold(n_splits=5).split(X_train_drop, y_train, groups=groups))

# %%  ── Model — TableVectorizer(cardinality_threshold=100) + HGBR ────────
# threshold=100 puts rater_id (50 unique), cohort and gene into the
# low-cardinality bucket → OneHotEncoder, never StringEncoder.

tv    = skrub.TableVectorizer(cardinality_threshold=100)
hgbr  = HistGradientBoostingRegressor(random_state=0)
model = Pipeline([("tablevectorizer", tv), ("histgradientboostingregressor", hgbr)])

# %%  ── OOF predictions on train ──────────────────────────────────────────
oof_pred = np.full(len(y_train), np.nan)
fold_rmses = []

for tr_idx, val_idx in cv_splits:
    m = clone(model)
    m.fit(X_train_s1.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = m.predict(X_train_s1.iloc[val_idx])
    oof_pred[val_idx] = preds
    fold_rmses.append(float(root_mean_squared_error(y_train.iloc[val_idx], preds)))

oof_mean = float(np.mean(fold_rmses))
oof_std  = float(np.std(fold_rmses))
print(f"Stage-1 OOF RMSE: {oof_mean:.4f} ± {oof_std:.4f}")

# %%  ── Test predictions (fit on all train) ──────────────────────────────
s1_full      = clone(model).fit(X_train_s1, y_train)
test_pred    = s1_full.predict(X_test_s1)

# %%  ── Save to data/ ─────────────────────────────────────────────────────
train_out = pd.DataFrame({"Index": X_train_raw.index, "stage1": oof_pred})
train_out.to_csv(DATA_DIR / "stage1_train.csv", index=False)

test_out = pd.DataFrame({"Index": X_test_raw.index, "stage1": test_pred})
test_out.to_csv(DATA_DIR / "stage1_test.csv", index=False)

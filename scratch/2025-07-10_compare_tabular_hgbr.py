"""Compare tabular_pipeline vs HGBR under patient-grouped CV — read-only probe."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate

import skrub

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
groups = X_raw["patient_id"]

X_full = X_raw.drop(columns=["patient_id"])
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

models = {
    "05_tabular": skrub.tabular_pipeline("regressor"),
    "04_hgbr":    HistGradientBoostingRegressor(random_state=0),
}

# 04_hgbr used only numeric columns
FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X_numeric = X_raw[FEATURE_COLS]
cv_splits_numeric = list(GroupKFold(n_splits=5).split(X_numeric, y, groups=groups))

print(f"{'Model':20s}  {'RMSE':>10s}  {'±':>6s}  {'MAE':>10s}")
print("-" * 55)

# tabular_pipeline on all columns
cv = cross_validate(
    models["05_tabular"], X_full, y, cv=cv_splits,
    scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
)
print(f"{'05_tabular':20s}  {-cv['test_rmse'].mean():10.4f}  {cv['test_rmse'].std():6.4f}  {-cv['test_mae'].mean():10.4f}")

# HGBR on numeric-only (experiment 04 reference)
cv2 = cross_validate(
    models["04_hgbr"], X_numeric, y, cv=cv_splits_numeric,
    scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
)
print(f"{'04_hgbr (numeric)':20s}  {-cv2['test_rmse'].mean():10.4f}  {cv2['test_rmse'].std():6.4f}  {-cv2['test_mae'].mean():10.4f}")

rmse_tab = -cv["test_rmse"].mean()
rmse_hgbr = -cv2["test_rmse"].mean()
print(f"\ntabular_pipeline vs HGBR: {(rmse_hgbr - rmse_tab) / rmse_hgbr * 100:+.1f}%")

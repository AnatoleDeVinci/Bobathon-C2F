"""Compare HGBR vs Ridge under patient-grouped CV — read-only probe."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]

X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X = X_raw[FEATURE_COLS]
groups = X_raw["patient_id"]

cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

models = {
    "hgbr":  HistGradientBoostingRegressor(random_state=0),
    "ridge": make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)),
}

print(f"{'Model':8s}  {'RMSE':>10s}  {'±':>6s}  {'MAE':>10s}")
print("-" * 42)
for name, model in models.items():
    cv = cross_validate(
        model, X, y, cv=cv_splits,
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
    )
    rmse_mean = -cv["test_rmse"].mean()
    rmse_std  =  cv["test_rmse"].std()
    mae_mean  = -cv["test_mae"].mean()
    print(f"{name:8s}  {rmse_mean:10.4f}  {rmse_std:6.4f}  {mae_mean:10.4f}")

print()
print("--- Prior experiments for reference (grouped CV) ---")
print("dummy     RMSE=16.4992  ±0.3067")
print("ridge     RMSE=10.5965  ±0.2388  (experiment 03)")

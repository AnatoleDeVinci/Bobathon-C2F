"""Compute grouped-CV RMSE for dummy and ridge — read-only probe."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
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

for name, model in [
    ("dummy", DummyRegressor(strategy="mean")),
    ("ridge", make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))),
]:
    cv_results = cross_validate(
        model, X, y, cv=cv_splits,
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
    )
    rmse_mean = -cv_results["test_rmse"].mean()
    rmse_std  = cv_results["test_rmse"].std()
    mae_mean  = -cv_results["test_mae"].mean()
    print(f"{name:8s}  RMSE={rmse_mean:.4f} ± {rmse_std:.4f}  MAE={mae_mean:.4f}")

# Also print the row-holdout numbers for comparison
print()
print("--- Row holdout (experiments 01/02) for reference ---")
print("dummy     RMSE=16.5721  MAE=13.6195")
print("ridge     RMSE=10.6657  MAE= 8.5677")

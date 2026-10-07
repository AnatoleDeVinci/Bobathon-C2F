"""Compare 01_dummy vs 02_ridge metrics on the same 80/20 holdout.

Read-only scratch probe — no evaluate or put calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
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

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_tr, y_tr)
p_dummy = dummy.predict(X_te)

ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
ridge.fit(X_tr, y_tr)
p_ridge = ridge.predict(X_te)

for name, preds in [("dummy", p_dummy), ("ridge", p_ridge)]:
    rmse = root_mean_squared_error(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    print(f"{name:8s}  RMSE={rmse:.4f}  MAE={mae:.4f}")

rmse_d = root_mean_squared_error(y_te, p_dummy)
rmse_r = root_mean_squared_error(y_te, p_ridge)
print(f"\nRidge RMSE improvement vs dummy: {(rmse_d - rmse_r) / rmse_d * 100:.1f}%")

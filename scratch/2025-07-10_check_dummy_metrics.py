"""Read metrics from the 01_dummy report via the skore hub project.

Scratch probe — read-only against the Project. Never calls evaluate or put.
"""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor

import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]

# Load to compute holdout metrics locally (report is on hub, harder to read back)
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_train = X_train_raw[FEATURE_COLS]

# Reproduce the 80/20 holdout used in the experiment
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np

X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_tr, y_tr)
preds = dummy.predict(X_te)

rmse = root_mean_squared_error(y_te, preds)
mae = mean_absolute_error(y_te, preds)
print(f"Holdout RMSE: {rmse:.4f}")
print(f"Holdout MAE:  {mae:.4f}")
print(f"Target std (train): {y_tr.std():.4f}")
print(f"Target mean (train): {y_tr.mean():.4f}")
print(f"Target mean (test):  {y_te.mean():.4f}")

# %% [markdown]
# # Experiment: 03_grouped_cv — Dummy & Ridge with patient-grouped CV
#
# **Date:** 2025-07-10
# **Goal:** Re-evaluate dummy and ridge with GroupKFold(5) on patient_id so that
#   whole patients are held out, matching the Kaggle evaluation protocol.
#   Results are pushed as two separate CrossValidationReports:
#   `03_grouped_cv_dummy` and `03_grouped_cv_ridge`.
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## Data

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X_train = X_train_raw[FEATURE_COLS]
groups = X_train_raw["patient_id"]

# %% [markdown]
# ## Grouped CV splits
#
# Precompute the index pairs so skore's sklearn path receives a plain list —
# skore calls `splitter.split(X, y)` without `groups=`, so we materialise
# the split once and pass the list directly.

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))
print(f"Folds: {len(cv_splits)}, "
      f"train sizes: {[len(tr) for tr, _ in cv_splits]}, "
      f"test sizes: {[len(te) for _, te in cv_splits]}")

# %% [markdown]
# ## Models

# %%
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate — grouped CV for dummy

# %%
report_dummy = skore.evaluate(dummy, X_train, y_train, splitter=cv_splits)
report_dummy

# %% [markdown]
# ## Evaluate — grouped CV for Ridge

# %%
report_ridge = skore.evaluate(ridge, X_train, y_train, splitter=cv_splits)
report_ridge

# %% [markdown]
# ## Project — push both reports to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

project.put("03_grouped_cv_dummy", report_dummy)
project.put("03_grouped_cv_ridge", report_ridge)

# %% [markdown]
# # Experiment: 04_hgbr — HistGradientBoosting vs Ridge (patient-grouped CV)
#
# **Date:** 2025-07-10
# **Goal:** Test whether HistGradientBoostingRegressor — which routes NaN values
#   down a learned branch rather than imputing them — beats Ridge under the same
#   patient-grouped GroupKFold(5) evaluation.
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
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
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

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
# ## Grouped CV splits (same as experiment 03)

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# %% [markdown]
# ## Models

# %%
# HistGBR handles NaN natively — no SimpleImputer.
hgbr = HistGradientBoostingRegressor(random_state=0)

# Ridge baseline (re-evaluated on the same splits for a fair comparison).
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate — grouped CV for HGBR

# %%
report_hgbr = skore.evaluate(hgbr, X_train, y_train, splitter=cv_splits)
report_hgbr

# %% [markdown]
# ## Evaluate — grouped CV for Ridge (comparison baseline)

# %%
report_ridge = skore.evaluate(ridge, X_train, y_train, splitter=cv_splits)
report_ridge

# %% [markdown]
# ## Project — push both reports to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

project.put("04_hgbr", report_hgbr)
project.put("04_hgbr_ridge", report_ridge)

# %% [markdown]
# ## Submit — Kaggle submission file for HGBR
#
# Refit HGBR on all training rows (no CV split), predict the test visits,
# write `submissions/04_hgbr.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = clone(hgbr).fit(X_train, y_train)

X_test_features = X_test[FEATURE_COLS]
predictions = final.predict(X_test_features)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "04_hgbr.csv", index=False)
print(f"Submission written: {submissions_dir / '04_hgbr.csv'} ({len(submission)} rows)")

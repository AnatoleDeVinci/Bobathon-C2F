# %% [markdown]
# # Experiment: 02_ridge — Ridge linear model vs dummy baseline
#
# **Date:** 2025-07-10
# **Goal:** Test whether numeric clinical features carry linear signal by comparing
#   `make_pipeline(SimpleImputer, Ridge)` against the dummy mean baseline.
#   Both models are evaluated side-by-side in a single ComparisonReport pushed to hub.
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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

# Numeric columns only — Ridge cannot handle strings or NaN without preprocessing.
# Identifiers (patient_id, rater_id) and string categoricals (cohort, gene) excluded.
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

# %% [markdown]
# ## Models

# %%
dummy = DummyRegressor(strategy="mean")

ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate — individual reports

# %%
# Ridge report — pushed to the hub and used for comparison.
report_ridge = skore.evaluate(ridge, X_train, y_train, splitter=0.2)
report_ridge

# %%
# Dummy report — evaluated on the same split for a fair side-by-side comparison.
report_dummy = skore.evaluate(dummy, X_train, y_train, splitter=0.2)

# %% [markdown]
# ## Project — push Ridge report to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", report_ridge)

# %% [markdown]
# ## Submit — Kaggle submission file for Ridge
#
# Refit Ridge on all training rows (no CV split), predict the test visits,
# write `submissions/02_ridge.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = clone(ridge).fit(X_train, y_train)

X_test_features = X_test[FEATURE_COLS]
predictions = final.predict(X_test_features)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print(f"Submission written: {submissions_dir / '02_ridge.csv'} ({len(submission)} rows)")

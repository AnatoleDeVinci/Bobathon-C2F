# %% [markdown]
# # Experiment: 01_dummy — Dummy mean baseline
#
# **Date:** 2025-07-10
# **Goal:** Establish the error floor for the Parkinson's UPDRS motor score regression.
#   `DummyRegressor(strategy="mean")` always predicts the training-set mean;
#   its RMSE ≈ std(target) and sets the minimum bar every real model must beat.
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor

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

# Feature columns: drop identifiers that should not be fed directly to the model.
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
# ## Evaluate

# %%
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X_train, y_train, splitter=0.2)
report

# %% [markdown]
# ## Project — push to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# %% [markdown]
# ## Submit — Kaggle submission file
#
# Refit on all training rows (no CV split), predict the test visits,
# write `submissions/01_dummy.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = DummyRegressor(strategy="mean")
final.fit(X_train, y_train)

X_test_features = X_test[FEATURE_COLS]
predictions = final.predict(X_test_features)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
print(f"Submission written: {submissions_dir / '01_dummy.csv'} ({len(submission)} rows)")

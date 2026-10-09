# %% [markdown]
# # Experiment: 06_patient_features — tabular_pipeline + patient-level features
#
# **Date:** 2025-07-10
# **Goal:** Test whether stateless per-patient aggregation and lag/lead features
#   (disease duration, visit ordering, off/on/ledd statistics, deviations,
#   ON−OFF gap, prev/next visit scores) improve over the raw tabular_pipeline
#   baseline (RMSE 7.51 ± 0.12, GroupKFold-5).
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import skrub

import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.features import add_patient_features
from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## Data
#
# Features are computed separately on train and test frames — no information
# flows across the split boundary.  `patient_id` and `Index` are kept until
# GroupKFold groups are extracted, then dropped before the model.

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

# Apply patient features independently on each frame.
X_train_feat = add_patient_features(X_train_raw)
X_test_feat = add_patient_features(X_test_raw)

# Extract groups before dropping patient_id.
groups = X_train_feat["patient_id"]

DROP_COLS = ["patient_id"]
X_train = X_train_feat.drop(columns=DROP_COLS)
X_test = X_test_feat.drop(columns=DROP_COLS)

# %% [markdown]
# ## Grouped CV splits (same as experiments 03–05)

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# %% [markdown]
# ## Model
#
# `tabular_pipeline("regressor")` handles NaN, mixed types, and the newly added
# numeric columns automatically.  No hyperparameter changes from experiment 05.

# %%
model = skrub.tabular_pipeline("regressor")

# %% [markdown]
# ## Evaluate — grouped CV

# %%
report = skore.evaluate(model, X_train, y_train, splitter=cv_splits)
report

# %% [markdown]
# ## Project — push to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("06_patient_features", report)

# %% [markdown]
# ## Submit — Kaggle submission file
#
# Refit on all training rows with patient features, predict the test visits,
# write `submissions/06_patient_features.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = clone(model).fit(X_train, y_train)
predictions = final.predict(X_test)

submission = X_test_raw.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "06_patient_features.csv", index=False)
print(
    f"Submission written: {submissions_dir / '06_patient_features.csv'}"
    f" ({len(submission)} rows)"
)

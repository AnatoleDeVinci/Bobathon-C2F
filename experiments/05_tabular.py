# %% [markdown]
# # Experiment: 05_tabular — skrub tabular_pipeline on all columns
#
# **Date:** 2025-07-10
# **Goal:** Test whether adding the string categorical columns (`gene`, `cohort`,
#   `rater_id`) via `skrub.tabular_pipeline("regressor")` improves over the
#   numeric-only HGBR baseline (RMSE 7.76 ± 0.14, GroupKFold-5).
# **Result:** filled in after the run.

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import skrub

import skore
from skore import Project, login

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## Data
#
# All columns except identifiers (Index, patient_id) and the target.
# tabular_pipeline handles NaN and mixed types automatically.

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

DROP_COLS = ["patient_id"]
X_train = X_train_raw.drop(columns=DROP_COLS)
groups = X_train_raw["patient_id"]

X_test_features = X_test.drop(columns=DROP_COLS, errors="ignore")

# %% [markdown]
# ## Grouped CV splits (same as experiments 03 and 04)

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# %% [markdown]
# ## Model
#
# `tabular_pipeline("regressor")` wires TableVectorizer + HistGradientBoostingRegressor.
# Automatically picks one-hot for low-cardinality strings (cohort, sexM),
# StringEncoder for higher-cardinality strings (gene, rater_id).

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
project.put("05_tabular", report)

# %% [markdown]
# ## Submit — Kaggle submission file
#
# Refit on all training rows, predict the test visits,
# write `submissions/05_tabular.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = clone(model).fit(X_train, y_train)
predictions = final.predict(X_test_features)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "05_tabular.csv", index=False)
print(f"Submission written: {submissions_dir / '05_tabular.csv'} ({len(submission)} rows)")

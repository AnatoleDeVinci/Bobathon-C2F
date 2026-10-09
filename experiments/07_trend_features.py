# %% [markdown]
# # Experiment: 07_trend_features — tabular_pipeline + trend, neighbour,
# interpolation, rolling, estimated-off and temporal features
#
# **Date:** 2025-07-10
# **Goal:** Test whether adding per-patient linear trends of off/on vs age,
#   extended neighbours (±2 visits), linear interpolation, rolling means,
#   estimated off for missing values, years_since_first_visit, and age_range
#   improves over experiment 06 (patient aggregation baseline).
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

X_train_feat = add_patient_features(X_train_raw)
X_test_feat = add_patient_features(X_test_raw)

groups = X_train_feat["patient_id"]

DROP_COLS = ["patient_id"]
X_train = X_train_feat.drop(columns=DROP_COLS)
X_test = X_test_feat.drop(columns=DROP_COLS)

# %% [markdown]
# ## Grouped CV splits (same as experiments 03–06)

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# %% [markdown]
# ## Model
#
# `tabular_pipeline("regressor")` handles NaN and mixed types automatically.
# No hyperparameter changes from experiment 06.

# %%
model = skrub.tabular_pipeline("regressor")

# %% [markdown]
# ## Evaluate — grouped CV

# %%
report = skore.evaluate(model, X_train, y_train, splitter=cv_splits)
report

# %% [markdown]
# ## RMSE comparison vs 06_patient_features

# %%
summary = report.metrics.summarize().frame()
print("07_trend_features CV metrics:")
print(summary.to_string())
print()

col_mean = [c for c in summary.columns if "mean" in c][0]
col_std  = [c for c in summary.columns if "std"  in c][0]
rmse_mean = float(summary.loc["rmse", col_mean])
rmse_std  = float(summary.loc["rmse", col_std])
print(f"07_trend_features   RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
print("06_patient_features RMSE: see hub report 06_patient_features for comparison")

# %% [markdown]
# ## Project — push to hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07_trend_features", report)

# %% [markdown]
# ## Submit — Kaggle submission file
#
# Refit on all training rows with the extended feature set, predict the test
# visits, write `submissions/07_trend_features.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = clone(model).fit(X_train, y_train)
predictions = final.predict(X_test)

submission = X_test_raw.reset_index()[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "07_trend_features.csv", index=False)
print(
    f"Submission written: {submissions_dir / '07_trend_features.csv'}"
    f" ({len(submission)} rows)"
)

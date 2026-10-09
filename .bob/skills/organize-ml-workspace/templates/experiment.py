# %% [markdown]
# # Experiment: <short title>
#
# **Date:** YYYY-MM-DD
# **Goal:** what hypothesis or change this experiment is testing.
# **Result:** filled in after the run.

# %%
import skore

from <pkg> import PROJECT_ROOT
from <pkg>.data import load_dataset, load_test_dataset
from <pkg>.evaluate import splitter
from <pkg>.pipeline import build_learner

# %% [markdown]
# ## Paths
#
# `PROJECT_ROOT` comes from the package's `__init__.py` and resolves
# from `__file__` — independent of the current working directory.
# Replace `"data"` with the project's actual data folder if different;
# data layout is user-owned. The same absolute path is passed both as
# the pipeline preview and via `data=` to `skore.evaluate`, so
# `learner.skb.preview()` and `skore.evaluate(...)` see the same
# binding.

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project
#
# Open the project that stores this experiment's report under a stable
# key (the file stem). All reports for this workspace live together so
# they can be compared across experiments.

# %%
# <SKORE_PROJECT_INIT>
project = skore.Project(
    name="<project-name>",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

# %% [markdown]
# ## Data and learner
#
# `data_dir_preview=DATA_DIR` makes `learner.skb.preview()` work; it
# does not affect what `skore.evaluate` actually fits on (that comes
# from `data=` below).

# %%
X, y = load_dataset()
learner = build_learner(data_dir_preview=DATA_DIR)

# %% [markdown]
# ## Evaluate
#
# Cross-validator and any metric overrides are imported from
# `<pkg>.evaluate`. `SkrubLearner.fit` takes a single environment dict
# (it does *not* implement `fit(X, y)`), so we pass the bindings via
# `data=`. Use
# the source-bound form (`data={"data_dir": str(DATA_DIR)}`) when the
# pipeline binds a source identifier; use `data={"X": X, "y": y}` for
# materialized bindings.

# %%
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)}, splitter=splitter)
report

# %% [markdown]
# ## Persist
#
# Key = file stem. Reusing this key in a future run overwrites the
# stored report — fork into a new experiment file if you want both.

# %%
project.put("<experiment-key>", report)

# %% [markdown]
# ## Submit
#
# `evaluate` only scores on held-out folds. The Kaggle file needs the
# learner refit on **all** training rows, then predicted on the test
# visits, and written to `submissions/<experiment-key>.csv` with the
# header `Index,target`. The filename is this script's stem (which
# names the estimator), so a later experiment never overwrites an
# earlier submission.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

# Test features use the same loader and column choices as training.
# `Index` is the identifier column the competition expects first.
X_test = load_test_dataset()

# Refit on the full training binding - no CV split here. The source
# var name matches the `skore.evaluate` call above. For a bare sklearn
# estimator, use `clone(estimator).fit(X, y)` and predict on `X_test`
# instead of an env-dict.
final = build_learner()
final.fit({"data_dir": str(DATA_DIR)})
final_predictions = final.predict({"data_dir": str(DATA_DIR)})

submission = X_test[["Index"]].copy()
submission["target"] = final_predictions
submission.to_csv(submissions_dir / "<experiment-key>.csv", index=False)

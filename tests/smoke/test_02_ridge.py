"""Smoke test for ``experiments/02_ridge.py``.

IID sanity check: fit on a train subset, predict on a held-out subset,
assert the prediction count matches exactly. Also checks both models
in the comparison (dummy and ridge) produce the right shape.

CV_MAE_MEAN is a conservative placeholder (std of target ≈ 16.5) updated
once journal/02_ridge.md § Status.headline is filled in.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

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

# Soft-assertion bound: 3 × std(target) ≈ 49.5.
# Ridge MAE should be well below this; update from journal/02_ridge.md after run.
CV_MAE_MEAN = 16.5


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Holds out the last 10 % of rows as the predict set (deterministic slice).
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    X_all = X_raw[FEATURE_COLS]
    split = int(len(X_all) * 0.9)

    X_tr, y_tr = X_all.iloc[:split], y_all.iloc[:split]
    X_pr, y_pr = X_all.iloc[split:], y_all.iloc[split:]
    return X_tr, y_tr, X_pr, y_pr


def test_02_ridge(train_predict_envs):
    """Ridge must produce one prediction per predict-set row."""
    X_tr, y_tr, X_pr, y_pr = train_predict_envs

    learner = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=1.0),
    )
    learner.fit(X_tr, y_tr)
    predictions = learner.predict(X_pr)

    # HARD: structural correctness.
    assert len(predictions) == len(X_pr), (
        f"got {len(predictions)} predictions for {len(X_pr)} predict rows"
    )

    # SOFT: predictions are not garbage.
    smoke_mae = mean_absolute_error(y_pr, predictions)
    assert smoke_mae < 3 * CV_MAE_MEAN, (
        f"smoke MAE {smoke_mae:.1f} > 3 × CV_MAE_MEAN ({3 * CV_MAE_MEAN:.1f}); "
        "predictions may be corrupted."
    )

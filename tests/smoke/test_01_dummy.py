"""Smoke test for ``experiments/01_dummy.py``.

The dummy regressor has no cross-row dependencies, so this is an IID
sanity check: fit on a train subset, predict on a held-out subset,
assert the prediction count matches exactly.

The soft assertion is deliberately loose (3 × std(target) ≈ 3 × 16.5 ≈ 50)
because the dummy always returns the training-mean constant — its MAE is
exactly |mean_train - mean_test|, which is small when the holdout is random.
CV_MAE_MEAN is hardcoded from the EDA target std as a conservative upper bound
(updated once the real CV result lands in journal/01_dummy.md).
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

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

# Conservative upper bound: 3 × std(target) ≈ 3 × 16.5 = 49.5.
# The dummy's MAE is |mean_train − mean_test|, which is near zero for
# a random split — this bound will never be approached in practice.
# Update with the real CV mean MAE once journal/01_dummy.md § Status.headline
# is filled in.
CV_MAE_MEAN = 16.5  # std(target) from EDA, used as placeholder


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Hold out the last 10 % of rows as the predict set (deterministic slice,
    no shuffling so the test is reproducible without a random seed).
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    X_all = X_raw[FEATURE_COLS]
    split = int(len(X_all) * 0.9)

    X_tr, y_tr = X_all.iloc[:split], y_all.iloc[:split]
    X_pr, y_pr = X_all.iloc[split:], y_all.iloc[split:]
    return X_tr, y_tr, X_pr, y_pr


def test_01_dummy(train_predict_envs):
    """Predict-time output must have one row per predict-set row."""
    X_tr, y_tr, X_pr, y_pr = train_predict_envs

    learner = DummyRegressor(strategy="mean")
    learner.fit(X_tr, y_tr)
    predictions = learner.predict(X_pr)

    # HARD: structural correctness — one prediction per row.
    assert len(predictions) == len(X_pr), (
        f"got {len(predictions)} predictions for {len(X_pr)} predict rows"
    )

    # SOFT: predictions are not garbage.
    smoke_mae = mean_absolute_error(y_pr, predictions)
    assert smoke_mae < 3 * CV_MAE_MEAN, (
        f"smoke MAE {smoke_mae:.1f} > 3 × CV_MAE_MEAN ({3 * CV_MAE_MEAN:.1f}); "
        "predictions may be corrupted."
    )

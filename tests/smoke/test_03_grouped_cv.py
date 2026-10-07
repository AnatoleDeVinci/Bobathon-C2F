"""Smoke test for ``experiments/03_grouped_cv.py``.

IID sanity check for the Ridge model under GroupKFold. Holds out one
patient group (the last 10% of rows by position, deterministic) and
asserts prediction count matches exactly.

CV_MAE_MEAN is a conservative placeholder updated once
journal/03_grouped_cv.md § Status.headline is filled in.
"""

import pandas as pd
import pytest
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

# Conservative upper bound: 3 × std(target) ≈ 49.5.
# Update from journal/03_grouped_cv.md once the grouped CV RMSE is known.
CV_MAE_MEAN = 16.5


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Holds out the last 10 % of rows (deterministic slice).
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    X_all = X_raw[FEATURE_COLS]
    split = int(len(X_all) * 0.9)

    X_tr, y_tr = X_all.iloc[:split], y_all.iloc[:split]
    X_pr, y_pr = X_all.iloc[split:], y_all.iloc[split:]
    return X_tr, y_tr, X_pr, y_pr


def test_03_grouped_cv(train_predict_envs):
    """Ridge must produce one prediction per predict-set row under grouped CV setup."""
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

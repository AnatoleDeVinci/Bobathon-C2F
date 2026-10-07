"""Smoke test for ``experiments/04_hgbr.py``.

IID sanity check: fit HGBR on a train subset, predict on a held-out subset
(no imputation — NaN passed through directly), assert the prediction count
matches exactly.

CV_MAE_MEAN is a conservative placeholder updated once
journal/04_hgbr.md § Status.headline is filled in.
"""

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
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

# Conservative upper bound: 3 × Ridge CV RMSE ≈ 3 × 10.6 = 31.8.
# Update from journal/04_hgbr.md once the grouped CV RMSE is known.
CV_MAE_MEAN = 10.6


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Holds out the last 10 % of rows (deterministic slice). NaN values
    are left as-is — HistGBR handles them natively.
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    X_all = X_raw[FEATURE_COLS]
    split = int(len(X_all) * 0.9)

    X_tr, y_tr = X_all.iloc[:split], y_all.iloc[:split]
    X_pr, y_pr = X_all.iloc[split:], y_all.iloc[split:]
    return X_tr, y_tr, X_pr, y_pr


def test_04_hgbr(train_predict_envs):
    """HGBR must produce one prediction per predict-set row (NaN inputs allowed)."""
    X_tr, y_tr, X_pr, y_pr = train_predict_envs

    learner = HistGradientBoostingRegressor(random_state=0)
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

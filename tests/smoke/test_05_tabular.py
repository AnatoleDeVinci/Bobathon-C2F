"""Smoke test for ``experiments/05_tabular.py``.

IID sanity check: fit skrub.tabular_pipeline on a train subset, predict on a
held-out subset (all columns except patient_id), assert prediction count matches.

CV_MAE_MEAN is set conservatively from experiment 04 (HGBR MAE 6.10).
Update from journal/05_tabular.md once the grouped CV result is in.
"""

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

import skrub

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
DROP_COLS = ["patient_id"]

# Conservative upper bound: 3 × HGBR grouped-CV MAE ≈ 3 × 6.1 = 18.3.
CV_MAE_MEAN = 6.1


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Holds out the last 10 % of rows (deterministic slice).
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    X_all = X_raw.drop(columns=DROP_COLS)
    split = int(len(X_all) * 0.9)

    X_tr, y_tr = X_all.iloc[:split], y_all.iloc[:split]
    X_pr, y_pr = X_all.iloc[split:], y_all.iloc[split:]
    return X_tr, y_tr, X_pr, y_pr


def test_05_tabular(train_predict_envs):
    """tabular_pipeline must produce one prediction per predict-set row."""
    X_tr, y_tr, X_pr, y_pr = train_predict_envs

    learner = skrub.tabular_pipeline("regressor")
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

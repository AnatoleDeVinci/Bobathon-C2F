"""Smoke test for ``experiments/06_patient_features.py``.

Fit skrub.tabular_pipeline on a patient-feature-enriched train subset,
predict on a disjoint subset (no shared patients), assert prediction count
matches and MAE is not worse than 3 × the 05_tabular grouped-CV MAE.

The train/predict split is patient-disjoint: the last 10 % of unique
patient_ids form the predict group.  This exercises the feature function's
behaviour on a smaller patient set (lower n_visits, noisier per-patient stats)
— the same regime the test CSV will be in at submission time.
"""

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

import skrub

from parkinson import PROJECT_ROOT
from parkinson.features import add_patient_features

DATA_DIR = PROJECT_ROOT / "data"
DROP_COLS = ["patient_id"]

# Conservative upper bound: 3 × tabular grouped-CV MAE ≈ 3 × 6.0 = 18.0.
CV_MAE_MEAN = 6.0


@pytest.fixture
def train_predict_envs():
    """Return (X_train, y_train, X_predict, y_predict).

    Split is patient-disjoint: last 10 % of unique patient_ids go to predict.
    Features are computed *separately* on each frame to mirror the experiment.
    """
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    patients = X_raw["patient_id"].unique()
    split_idx = int(len(patients) * 0.9)
    train_patients = set(patients[:split_idx])
    predict_patients = set(patients[split_idx:])

    tr_mask = X_raw["patient_id"].isin(train_patients)
    pr_mask = X_raw["patient_id"].isin(predict_patients)

    # Compute features independently on each subset.
    X_tr_feat = add_patient_features(X_raw[tr_mask])
    X_pr_feat = add_patient_features(X_raw[pr_mask])

    y_tr = y_all[tr_mask]
    y_pr = y_all[pr_mask]

    return (
        X_tr_feat.drop(columns=DROP_COLS),
        y_tr,
        X_pr_feat.drop(columns=DROP_COLS),
        y_pr,
    )


def test_06_patient_features(train_predict_envs):
    """tabular_pipeline with patient features must produce one prediction per row."""
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

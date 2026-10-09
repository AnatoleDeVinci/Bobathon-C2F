"""Smoke test for ``experiments/09_tuned.py``.

Patient-disjoint split: last 10 % of unique patient_ids → predict set.
Builds the stage-2 frame (same logic as 08/09), fits a single HGBR with
the representative best hyperparameters, and asserts structural + soft-MAE
correctness.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import skrub

from parkinson import PROJECT_ROOT
from parkinson.features import _linear_trend, add_patient_features

DATA_DIR      = PROJECT_ROOT / "data"
DROP_COLS     = ["patient_id"]
CV_MAE_MEAN   = 4.30  # conservative upper bound from 08_stack


def add_stage1_meta(df, s1_col="s1_pred"):
    out = df.copy()
    for k in (1, 2):
        out[f"s1_prev{k}"] = out.groupby("patient_id")[s1_col].shift(k)
        out[f"s1_next{k}"] = out.groupby("patient_id")[s1_col].shift(-k)
    grp = out.groupby("patient_id")[s1_col]
    out["s1_mean"]   = grp.transform("mean")
    out["s1_median"] = grp.transform("median")
    out["s1_std"]    = grp.transform("std")
    slopes, intercepts = {}, {}
    for pid, g in out.groupby("patient_id"):
        s, i = _linear_trend(g[s1_col], g["age"])
        slopes[pid] = s
        intercepts[pid] = i
    out["s1_trend_slope"]  = out["patient_id"].map(slopes)
    out["s1_trend_fitted"] = (
        out["s1_trend_slope"] * out["age"] + out["patient_id"].map(intercepts)
    )
    return out


_S2_DROP_STRING = ["cohort", "gene", "rater_id"]


def build_stage2_frame(X07, patient_ids, ages, s1_predictions):
    frame = X07.drop(columns=_S2_DROP_STRING, errors="ignore").copy()
    frame["patient_id"] = patient_ids.values
    frame["age_raw"]    = ages.values
    frame["s1_pred"]    = s1_predictions
    frame = frame.sort_values(["patient_id", "age_raw"])
    frame = add_stage1_meta(frame, s1_col="s1_pred")
    frame = frame.drop(columns=["patient_id", "age_raw"])
    frame = frame.sort_index()
    return frame


@pytest.fixture
def envs():
    X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_all = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]

    patients  = X_raw["patient_id"].unique()
    split_idx = int(len(patients) * 0.9)
    tr_pats   = set(patients[:split_idx])
    pr_pats   = set(patients[split_idx:])

    tr_mask = X_raw["patient_id"].isin(tr_pats)
    pr_mask = X_raw["patient_id"].isin(pr_pats)

    X_tr_feat = add_patient_features(X_raw[tr_mask])
    X_pr_feat = add_patient_features(X_raw[pr_mask])
    X_tr07    = X_tr_feat.drop(columns=DROP_COLS)
    X_pr07    = X_pr_feat.drop(columns=DROP_COLS)
    y_tr      = y_all[tr_mask]
    y_pr      = y_all[pr_mask]

    s1 = skrub.tabular_pipeline("regressor").fit(X_tr07, y_tr)
    X_tr_s2 = build_stage2_frame(X_tr07, X_tr_feat["patient_id"], X_tr_feat["age"], s1.predict(X_tr07))
    X_pr_s2 = build_stage2_frame(X_pr07, X_pr_feat["patient_id"], X_pr_feat["age"], s1.predict(X_pr07))

    return X_tr_s2, y_tr, X_pr_s2, y_pr


def test_09_tuned(envs):
    """Tuned HGBR must produce one prediction per predict-set row."""
    X_tr, y_tr, X_pr, y_pr = envs

    learner = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=0,
    )
    learner.fit(X_tr, y_tr)
    predictions = learner.predict(X_pr)

    assert len(predictions) == len(X_pr), (
        f"got {len(predictions)} predictions for {len(X_pr)} predict rows"
    )

    mae = mean_absolute_error(y_pr, predictions)
    assert mae < 3 * CV_MAE_MEAN, (
        f"smoke MAE {mae:.1f} > 3 × CV_MAE_MEAN ({3 * CV_MAE_MEAN:.1f})"
    )

"""Supervised reference maps stay inside reseller boundaries and preserve ties."""

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from comparisons.crossfit_percentile import FEATURES, fit_components
from comparisons.retained_calibration import calibrate, mapped_components, score_logits


def test_calibration_is_reconstructed_and_ties_remain_tied():
    reference = np.array([0, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1])
    labels = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    held = np.array([0, 0.2, 0.2, 1])
    score, identity = calibrate(reference, labels, held)
    np.testing.assert_allclose(
        score,
        expit(score_logits(held).ravel() * identity["slope"] + identity["intercept"]),
    )
    assert score[1] == score[2]
    assert np.isfinite(score).all()
    assert identity["iterations"] < 1000


@pytest.mark.parametrize("bad", [[np.nan], [-0.01], [1.01], [np.inf]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError, match="Invalid"):
        score_logits(bad)


def test_single_class_reference_is_infeasible():
    with pytest.raises(ValueError, match="both binary"):
        calibrate([0.1, 0.2], [0, 0], [0.4])


def test_synthetic_component_smoke_enforces_fit_reference_and_held_isolation():
    train = pd.DataFrame({feature: np.ones(120) for feature in FEATURES})
    train["StoreID"] = np.repeat(range(1, 61), 2)
    train["snapshot"] = pd.to_datetime(["2013-01-01", "2013-04-01"] * 60)
    train["churn"] = [0, 1] * 60
    held = train.iloc[:6].drop(columns="churn").assign(StoreID=range(101, 107))
    _, records = fit_components(train, held)
    for record in records:
        record["stage"] = "outer"
    score, fitted = mapped_components(train, held[["StoreID", "snapshot"]], records)
    np.testing.assert_allclose(score, 0.5)
    assert len(fitted) == 10
    assert all(item["fitting_rows"] == item["reference_rows"] == 60 for item in fitted)
    damaged = copy.deepcopy(records)
    damaged[0]["training_sha256"] = "bad"
    with pytest.raises(ValueError, match="fitting hash"):
        mapped_components(train, held[["StoreID", "snapshot"]], damaged)
    with pytest.raises(ValueError, match="Held reseller"):
        mapped_components(train, train.iloc[:2][["StoreID", "snapshot"]], records)

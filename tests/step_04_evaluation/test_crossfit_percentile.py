"""Held-training-reference alignment preserves ties and reseller isolation."""

import numpy as np
import pandas as pd
import pytest

from comparisons.crossfit_percentile import (
    FEATURES,
    NORMALIZED,
    RAW,
    fit_components,
    percentile_scores,
)


def test_reference_midpoint_ties_and_endpoint_behavior():
    np.testing.assert_allclose(
        percentile_scores(
            np.array([0.1, 0.2, 0.2, 0.9]), np.array([0, 0.1, 0.2, 0.55, 0.9, 1])
        ),
        [0, 0.125, 0.5, 0.6875, 0.875, 1],
    )
    np.testing.assert_allclose(
        percentile_scores(np.array([0.3, 0.3]), np.array([0.2, 0.3, 0.3, 0.4])),
        [0, 0.5, 0.5, 1],
    )
    with pytest.raises(ValueError, match="Invalid"):
        percentile_scores(np.array([]), np.array([0.1]))


def test_crossfit_components_are_out_of_fit_and_do_not_receive_validation_labels():
    train = pd.DataFrame({feature: np.ones(120) for feature in FEATURES})
    train["StoreID"] = np.repeat(range(60), 2)
    train["snapshot"] = pd.to_datetime(["2013-01-01", "2013-04-01"] * 60)
    train["churn"] = [0, 1] * 60
    valid = train.iloc[:6].drop(columns="churn").assign(StoreID=range(100, 106))
    scores, identities = fit_components(train, valid)
    assert len(identities) == 10
    np.testing.assert_allclose(scores[RAW], 0.5)
    np.testing.assert_allclose(scores[NORMALIZED], 0.5)
    for row in identities:
        reference = {case["StoreID"] for case in row["reference_cases"]}
        assert len(reference) == 30
        assert reference.isdisjoint(valid.StoreID)
        assert len(row["reference_scores"]) == 60
        assert len(row["validation_raw_scores"]) == 6
        assert np.isfinite(row["validation_percentiles"]).all()

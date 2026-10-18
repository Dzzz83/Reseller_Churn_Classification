"""Training-prior normalization and fixed score pairing reject target leakage."""

import numpy as np
import pandas as pd
import pytest

from comparisons.continuation_blend import combine_rows, undo_balanced_odds


def test_weight_odds_inverse_endpoints_and_ties():
    prior = 0.15
    probability = np.array([0.0, 0.01, 0.15, 0.4, 0.4, 1.0])
    w1, w0 = 1 / (2 * prior), 1 / (2 * (1 - prior))
    weighted = probability * w1 / (probability * w1 + (1 - probability) * w0)
    np.testing.assert_allclose(
        undo_balanced_odds(weighted, prior), probability, atol=1e-15
    )
    assert (
        undo_balanced_odds(weighted, prior)[3] == undo_balanced_odds(weighted, prior)[4]
    )


@pytest.mark.parametrize(
    "prior,scores",
    [(0, [0.5]), (1, [0.5]), (np.nan, [0.5]), (0.2, [np.nan]), (0.2, [-1])],
)
def test_invalid_prior_or_score_rejected(prior, scores):
    with pytest.raises(ValueError, match="Invalid"):
        undo_balanced_odds(scores, prior)


def test_only_fitting_labels_set_prior_and_case_pairing_is_exact():
    train = pd.DataFrame({"StoreID": [1, 2, 3, 4], "churn": [1, 0, 0, 0]})
    a = pd.DataFrame(
        {
            "StoreID": [10, 11],
            "snapshot": pd.Timestamp("2013-04-01"),
            "churn": [0, 1],
            "score": [0.1, 0.4],
        }
    )
    b = a.assign(score=[0.2, 0.8]).iloc[::-1]
    combined = combine_rows(a, b, train)
    np.testing.assert_allclose(
        combined.blended_score,
        (a.score.to_numpy() + undo_balanced_odds([0.2, 0.8], 0.25)) / 2,
    )
    assert combined.fitting_prevalence.eq(0.25).all()
    changed = combine_rows(
        a.assign(churn=1 - a.churn), b.assign(churn=1 - b.churn), train
    )
    np.testing.assert_array_equal(changed.blended_score, combined.blended_score)
    with pytest.raises(ValueError, match="Unmatched"):
        combine_rows(a, b.iloc[:1], train)
    with pytest.raises(ValueError, match="prior leakage"):
        combine_rows(a, b, train.assign(StoreID=[10, 2, 3, 4]))

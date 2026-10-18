import numpy as np
import pandas as pd
import pytest

from reseller_churn.modeling.ensemble import apply_threshold, mean_probabilities


@pytest.mark.parametrize(
    "probabilities",
    [[[np.nan, np.nan]], [[-0.1, 1.1]], [[0.2, 0.2]], [[0.1, 0.2, 0.7]]],
)
def test_invalid_estimator_outputs_are_rejected(probabilities):
    class InvalidEstimator:
        def predict_proba(self, features):
            return probabilities

    with pytest.raises(ValueError):
        mean_probabilities([InvalidEstimator()], pd.DataFrame({"feature": [1]}))


def test_empty_ensemble_cannot_silently_produce_nan():
    with pytest.raises(ValueError, match="at least one"):
        mean_probabilities([], pd.DataFrame({"feature": [1]}))


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError):
        apply_threshold([0.5], threshold)

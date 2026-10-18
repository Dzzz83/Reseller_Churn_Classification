"""Weighting uses fitting rows only and retains the original class total scale."""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from comparisons.grouped_precision_recall import (
    BASE_PARAMETERS,
    FEATURES,
    synthetic_development,
)
from comparisons.reseller_weighting import fit_weighted, reseller_weights


def test_class_totals_and_inverse_history_ratios():
    train = pd.DataFrame({"StoreID": [1, 1, 1, 2, 3, 3], "churn": [0, 0, 1, 1, 0, 1]})
    weights = reseller_weights(train)
    for label in [0, 1]:
        assert np.isclose(weights[train.churn.eq(label)].sum(), 3.0)
    assert np.isclose(weights[3] / weights[2], 3.0)
    subset = train.iloc[[0, 2, 3]].copy()
    # Counts recomputed on this fitting subset: Store1 has two rows, not three.
    subset_weights = reseller_weights(subset)
    assert np.isclose(subset_weights[2] / subset_weights[1], 2.0)


def test_one_snapshot_per_reseller_equals_original_balanced_weights():
    train = pd.DataFrame({"StoreID": [1, 2, 3, 4, 5], "churn": [0, 0, 0, 1, 1]})
    np.testing.assert_array_equal(
        reseller_weights(train), compute_sample_weight("balanced", train.churn)
    )


def test_synthetic_weighted_fit_has_no_held_labels_and_five_seed_records():
    data = synthetic_development()
    fitting = data[data.StoreID.le(data.StoreID.median())]
    held = data[data.StoreID.gt(data.StoreID.median())][FEATURES]
    scores, identities, spread, _ = fit_weighted(
        fitting,
        held,
        dict(
            id="weighted",
            features=FEATURES,
            parameters={**BASE_PARAMETERS, "max_iter": 10},
        ),
    )
    assert len(identities) == 5 and len(scores) == len(held)
    assert np.isfinite(scores).all() and spread >= 0
    np.testing.assert_array_equal(identities[0]["weights"], reseller_weights(fitting))

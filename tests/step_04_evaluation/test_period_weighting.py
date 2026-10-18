"""Fixed cell weights use fitting counts and retain the balanced class scale."""

import numpy as np
import pandas as pd
import pytest

from comparisons.grouped_precision_recall import (
    BASE_PARAMETERS,
    FEATURES,
    SNAPSHOTS,
    synthetic_development,
)
from comparisons.period_weighting import fit_period_weighted, period_weights


def test_unequal_cells_each_have_one_eighth_of_total_weight():
    rows = []
    for index, snapshot in enumerate(SNAPSHOTS):
        for y in [0, 1]:
            rows.extend(
                [dict(snapshot=pd.Timestamp(snapshot), churn=y)] * (index + y + 1)
            )
    frame = pd.DataFrame(rows)
    weighted = frame.assign(weight=period_weights(frame))
    np.testing.assert_allclose(
        weighted.groupby(["snapshot", "churn"]).weight.sum(), len(frame) / 8
    )
    np.testing.assert_allclose(weighted.groupby("churn").weight.sum(), len(frame) / 2)
    subset = frame.drop(index=frame.index[-1])
    revised = subset.assign(weight=period_weights(subset))
    np.testing.assert_allclose(
        revised.groupby(["snapshot", "churn"]).weight.sum(), len(subset) / 8
    )


def test_missing_period_or_outcome_cell_is_infeasible():
    frame = pd.DataFrame(
        [dict(snapshot=pd.Timestamp(s), churn=y) for s in SNAPSHOTS for y in [0, 1]]
    )
    with pytest.raises(ValueError, match="Both fitting outcomes"):
        period_weights(frame.iloc[:-1])
    with pytest.raises(ValueError, match="All four"):
        period_weights(frame.iloc[:-2])


def test_synthetic_period_weighted_fit_retains_exact_weight_vector():
    data = synthetic_development()
    fitting = data[data.StoreID.le(data.StoreID.median())]
    held = data[data.StoreID.gt(data.StoreID.median())][FEATURES]
    score, identities, _, _ = fit_period_weighted(
        fitting,
        held,
        dict(
            id="period",
            features=FEATURES,
            parameters={**BASE_PARAMETERS, "max_iter": 10},
        ),
    )
    assert np.isfinite(score).all() and len(identities) == 5
    np.testing.assert_array_equal(identities[0]["weights"], period_weights(fitting))

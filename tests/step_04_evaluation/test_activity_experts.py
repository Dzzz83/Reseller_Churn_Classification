"""Fixed routing preserves all cases, original weights and outcome isolation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from comparisons.activity_experts import fit_experts
from comparisons.grouped_precision_recall import BASE_PARAMETERS, FEATURES


def synthetic():
    rng = np.random.default_rng(103)
    train = pd.DataFrame(rng.uniform(size=(160, len(FEATURES))), columns=FEATURES)
    train["StoreID"] = np.arange(1, 161)
    train["snapshot"] = pd.Timestamp("2013-01-01")
    train["n_orders_3m"] = np.arange(160) % 2
    train["churn"] = (np.arange(160) % 4 >= 2).astype(int)
    train.loc[train.n_orders_3m.eq(0), "weighted_discount_3m"] = np.nan
    return train


def test_both_routes_and_empty_held_route_keep_original_weights():
    train = synthetic()
    held = train.iloc[:10].drop(columns="churn").assign(StoreID=np.arange(1001, 1011))
    spec = {
        "id": "experts",
        "features": FEATURES,
        "parameters": {**BASE_PARAMETERS, "max_iter": 10},
    }
    score, records, _, _ = fit_experts(train, held, spec)
    assert np.isfinite(score).all() and len(score) == 10 and len(records) == 10
    weights = compute_sample_weight("balanced", train.churn)
    for record in records:
        route = train.n_orders_3m.gt(0).eq(record["recent_purchase"])
        np.testing.assert_array_equal(record["weights"], weights[route])
        assert len(record["validation_cases"]) == 5
        assert ("weighted_discount_3m" in record["omitted_all_missing"]) == (
            not record["recent_purchase"]
        )
    only_recent = held[held.n_orders_3m.gt(0)]
    score, records, _, _ = fit_experts(train, only_recent, spec)
    assert len(score) == 5
    assert all(not r["validation_cases"] for r in records if not r["recent_purchase"])


def test_missing_training_class_is_infeasible_without_fallback():
    train = synthetic()
    train.loc[train.n_orders_3m.eq(0), "churn"] = 0
    with pytest.raises(ValueError, match="requires both classes"):
        fit_experts(
            train,
            train.drop(columns="churn"),
            dict(id="experts", features=FEATURES, parameters=BASE_PARAMETERS),
        )

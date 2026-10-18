"""Feature formulas and the contracts between the two independent builders."""

import numpy as np
import pandas as pd
import pytest

from reseller_churn.config.frozen_model import (
    DISCOUNT_FEATURES,
    FEATURES,
    PURCHASE_FEATURES,
)
from reseller_churn.data.preparation import KEYS, build_features, load_sources
from reseller_churn.features.builders import (
    build_discount_features,
    build_purchase_features,
)


def test_purchase_builder_returns_only_its_features_without_changing_history():
    history = pd.DataFrame(
        {
            "StoreID": [1, 1, 1, 2],
            "SalesOrderID": [10, 11, 12, 13],
            "OrderDate": pd.to_datetime(
                ["2013-09-10", "2013-07-01", "2013-06-30", "2013-04-01"]
            ),
            "SubTotal": [100.0, 300.0, 100.0, 200.0],
            "rev_bikes": [60.0, 0.0, 0.0, 200.0],
            "rev_accessories": [0.0, 200.0, 0.0, 0.0],
            "rev_clothing": [0.0, 100.0, 0.0, 0.0],
        }
    )
    original_history = history.copy(deep=True)
    features = build_purchase_features(history, pd.Timestamp("2013-10-01"))
    assert list(features.columns) == ["StoreID", *PURCHASE_FEATURES]
    first = features.set_index("StoreID").loc[1]
    assert first.n_orders_3m == 2
    assert first.revenue_3m == 400.0
    assert first.revenue_12m == 500.0
    assert first.recency_days == 21
    assert first.share_bikes == pytest.approx(0.12)
    assert first.share_accessories == pytest.approx(0.4)
    assert first.share_clothing == pytest.approx(0.2)
    second = features.set_index("StoreID").loc[2]
    assert second.n_orders_3m == 0
    assert second.revenue_3m == 0.0
    pd.testing.assert_frame_equal(history, original_history)


def test_discount_builder_weights_gross_values_and_preserves_missing_periods():
    # Store 1 includes annual/recent starts, a previous-period purchase, and
    # purchases just outside the annual window. The two recent gross values
    # are 900 and 100, so their weighted discount is 0.28, not their mean 0.20.
    lines = pd.DataFrame(
        {
            "StoreID": [1, 1, 1, 1, 1, 1, 2, 3, 4],
            "OrderDate": pd.to_datetime(
                [
                    "2013-09-30",
                    "2013-07-01",
                    "2013-06-30",
                    "2012-10-01",
                    "2012-09-30",
                    "2013-10-01",
                    "2013-09-01",
                    "2013-04-01",
                    "2013-10-01",
                ]
            ),
            "OrderQty": [3, 1, 1, 1, 1, 1, 1, 1, 1],
            "UnitPrice": [
                300.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "UnitPriceDiscount": [0.3, 0.1, 0.1, 0.0, 1.0, 1.0, 0.0, 0.2, 0.5],
        }
    )
    original_lines = lines.copy(deep=True)
    features = build_discount_features(lines, pd.Timestamp("2013-10-01"))
    assert list(features.columns) == ["StoreID", *DISCOUNT_FEATURES]
    stores = features.set_index("StoreID")
    assert stores.index.tolist() == [1, 2, 3]
    assert stores.loc[1, "weighted_discount_3m"] == pytest.approx(0.28)
    assert stores.loc[1, "discount_change_3m"] == pytest.approx(0.18)
    assert stores.loc[1, "discounted_line_share_12m"] == pytest.approx(0.75)
    assert stores.loc[2, "weighted_discount_3m"] == 0.0
    assert stores.loc[2, "discounted_line_share_12m"] == 0.0
    assert np.isnan(stores.loc[2, "discount_change_3m"])
    assert np.isnan(stores.loc[3, "weighted_discount_3m"])
    assert np.isnan(stores.loc[3, "discount_change_3m"])
    assert stores.loc[3, "discounted_line_share_12m"] == 1.0
    pd.testing.assert_frame_equal(lines, original_lines)


def test_combination_keeps_eligible_store_when_discount_builder_has_no_row(
    source_paths, monkeypatch
):
    import reseller_churn.data.preparation as feature_module

    sources = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    original_builder = build_discount_features

    def without_store_two(lines, snapshot):
        discounts = original_builder(lines, snapshot)
        return discounts[discounts.StoreID.ne(2)]

    monkeypatch.setattr(feature_module, "build_discount_features", without_store_two)
    features = build_features(sources, "2013-10-01")
    store = features.set_index("StoreID").loc[2]
    assert store[list(DISCOUNT_FEATURES)].isna().all()
    assert store.revenue_12m > 0
    assert features.StoreID.is_unique
    assert features.StoreID.is_monotonic_increasing
    assert list(features.columns) == [*KEYS, *FEATURES]


@pytest.mark.parametrize("snapshot", ["2011-05-31", "2014-05-01"])
def test_no_eligible_stores_returns_empty_feature_table(source_paths, snapshot):
    sources = load_sources(source_paths, "2010-01-01", "2015-01-01", "2015-01-01")
    features = build_features(sources, snapshot)
    assert features.empty
    assert list(features.columns) == [*KEYS, *FEATURES]

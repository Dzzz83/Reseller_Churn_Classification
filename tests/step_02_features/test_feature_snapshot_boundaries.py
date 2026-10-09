"""Boundary checks for historical-only reseller features."""

import pandas as pd

from reseller_churn.features.feature_pipeline import FeaturePipeline


def test_future_orders_do_not_change_snapshot_features() -> None:
    snapshot = "2013-04-01"

    def order(order_id: int, store_id: int, date: str, revenue: float) -> dict:
        return {
            "SalesOrderID": order_id,
            "StoreID": store_id,
            "OrderDate": pd.Timestamp(date),
            "SubTotal": revenue,
            "rev_bikes": revenue,
            "rev_components": 0.0,
            "rev_clothing": 0.0,
            "rev_accessories": 0.0,
        }

    history = pd.DataFrame(
        [
            order(1, 1, "2012-06-01", 100.0),
            order(2, 1, "2012-11-01", 50.0),
            order(3, 1, "2013-03-31", 30.0),
        ]
    )
    future = pd.DataFrame(
        [
            order(4, 1, snapshot, 1000.0),  # Exact snapshot is excluded.
            order(5, 1, "2013-09-15", 2000.0),
            order(6, 2, snapshot, 900.0),  # Future-only store is ineligible.
        ]
    )
    stores = pd.DataFrame(
        {"StoreID": [1, 2], "YearOpened": [2000, 2005]}
    )

    pipeline = FeaturePipeline()
    historical_features = pipeline.build(history, stores, [snapshot])
    features_with_future_orders = pipeline.build(
        pd.concat([history, future], ignore_index=True),
        stores,
        [snapshot],
    )

    pd.testing.assert_frame_equal(
        historical_features,
        features_with_future_orders,
    )
    assert historical_features["StoreID"].tolist() == [1]

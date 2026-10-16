"""Synthetic guards for descriptive diagnosis; no historical outcomes read."""

import numpy as np
import pandas as pd
import pytest
from current_model_diagnosis import (
    classify,
    development_labels,
    select_cutoff,
    snapshot_features,
)
from report_development_diagnosis import read_saved_csv


def example():
    orders = pd.DataFrame(
        {
            "StoreID": [1, 1, 2],
            "SalesOrderID": [1, 2, 3],
            "OrderDate": pd.to_datetime(["2013-04-01", "2013-07-01", "2013-04-01"]),
            "SubTotal": [90.0, 80.0, 100.0],
            "rev_bikes": [90.0, 80.0, 100.0],
            "rev_accessories": [0.0, 0.0, 0.0],
            "rev_clothing": [0.0, 0.0, 0.0],
        }
    )
    details = orders[["StoreID", "SalesOrderID", "OrderDate"]].assign(
        OrderQty=1.0, UnitPrice=100.0, UnitPriceDiscount=[0.1, 0.2, 0.0]
    )
    return orders, details


def test_discount_windows_and_native_absence():
    orders, details = example()
    data = snapshot_features(orders, details, "2013-10-01").set_index("StoreID")
    assert data.loc[1, "weighted_discount_3m"] == pytest.approx(0.2)
    assert data.loc[1, "discount_change_3m"] == pytest.approx(0.1)
    assert np.isnan(data.loc[2, "weighted_discount_3m"])
    assert data.loc[2, "discounted_line_share_12m"] == 0.0
    assert data.loc[2, "n_orders_3m"] == 0
    assert data.loc[1, "revenue_ratio_3m"] == pytest.approx(80 / 90)


def test_future_orders_cannot_change_features():
    orders, details = example()
    expected = snapshot_features(orders, details, "2013-10-01")
    future_order = orders.iloc[[0]].assign(
        SalesOrderID=99, OrderDate=pd.Timestamp("2013-10-01"), SubTotal=9999.0
    )
    future_detail = details.iloc[[0]].assign(
        SalesOrderID=99, OrderDate=pd.Timestamp("2013-10-01")
    )
    actual = snapshot_features(
        pd.concat([orders, future_order]),
        pd.concat([details, future_detail]),
        "2013-10-01",
    )
    pd.testing.assert_frame_equal(expected, actual)


def test_development_labels_half_open_and_protected_dates_rejected():
    orders = pd.DataFrame(
        {
            "StoreID": [1, 2, 3],
            "OrderDate": pd.to_datetime(["2013-01-01", "2013-07-01", "2012-12-31"]),
        }
    )
    keys = pd.DataFrame({"StoreID": [1, 2, 3]})
    assert development_labels(orders, keys, "2013-01-01").churn.tolist() == [0, 1, 1]
    with pytest.raises(AssertionError, match="approved development"):
        development_labels(orders, keys, "2013-10-01")


def test_cutoff_selection_requires_strictly_over_90_recall():
    y = np.r_[np.ones(10), np.zeros(3)]
    scores = np.r_[np.linspace(0.1, 1.0, 10), [0.05, 0.3, 0.9]]
    threshold = select_cutoff(y, scores)
    assert threshold == pytest.approx(0.1)
    assert ((scores >= threshold) & (y == 1)).sum() == 10


def test_cutoff_equality_and_error_categories():
    data = pd.DataFrame(
        {
            "score": [0.06, 0.059, 0.1, 0.01],
            "churn": [1, 1, 0, 0],
            "n_orders_3m": [1, 1, 1, 1],
        }
    )
    assert classify(data).result.tolist() == [
        "caught_churn",
        "missed_churn",
        "false_alert",
        "correct_non_churn",
    ]


def test_saved_score_equal_to_selected_cutoff_preserves_decision(tmp_path):
    boundary = 0.12345678901234568
    path = tmp_path / "scores.csv"
    pd.DataFrame({"score": [boundary], "cutoff": [boundary]}).to_csv(path, index=False)
    loaded = read_saved_csv(path)
    assert loaded.score.item() == boundary
    assert loaded.score.ge(boundary).item()

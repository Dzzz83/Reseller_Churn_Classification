"""Temporal, denominator, and source-integrity checks for new order features."""

import numpy as np
import pandas as pd
import pytest

from build_companyx_features import build_features, clean_extracts


def example_history():
    headers = pd.DataFrame({
        "SalesOrderID": [1, 2, 3, 4, 5], "StoreID": [1] * 5,
        "OrderDate": pd.to_datetime([
            "2011-12-31", "2012-01-01", "2012-07-01", "2012-10-01", "2013-01-01",
        ]),
        "SalesPersonID": [10] * 5, "TerritoryID": [2] * 5,
        "SubTotal": [9990, 20, 40, 120, 9990],
    })
    lines = pd.DataFrame({
        "SalesOrderID": [1, 2, 3, 4, 5], "SalesOrderDetailID": [1, 2, 3, 4, 5],
        "ProductID": [99, 1, 1, 2, 99], "OrderQty": [999, 2, 4, 8, 999],
        "UnitPrice": [10, 10, 10, 20, 10],
        "UnitPriceDiscount": [0, 0, 0, .25, 0],
        "LineTotal": [9990, 20, 40, 120, 9990],
    })
    keys = pd.DataFrame({"StoreID": [1], "snapshot": [pd.Timestamp("2013-01-01")]})
    return keys, headers, lines


def test_features_respect_period_boundaries_and_product_novelty():
    keys, headers, lines = example_history()
    row = build_features(keys, headers, lines).iloc[0]
    assert row["avg_qty_3m"] == 8
    assert row["avg_qty_12m"] == pytest.approx(14 / 3)
    assert row["avg_lines_3m"] == 1
    assert row["qty_ratio_3m"] == 2
    assert row["net_unit_price_3m"] == 15
    assert row["weighted_discount_3m"] == .25
    assert row["discount_change_3m"] == .25
    assert row["discounted_line_share_12m"] == pytest.approx(1 / 3)
    assert row["unique_products_3m"] == 1
    assert row["unique_products_12m"] == 2
    assert row["new_product_share_3m"] == 1


def test_no_recent_orders_preserve_missing_averages_and_zero_totals():
    keys, headers, lines = example_history()
    headers = headers[headers.SalesOrderID.ne(4)]
    lines = lines[lines.SalesOrderID.ne(4)]
    row = build_features(keys, headers, lines).iloc[0]
    assert pd.isna(row["avg_qty_3m"])
    assert pd.isna(row["weighted_discount_3m"])
    assert pd.isna(row["new_product_share_3m"])
    assert row["unique_products_3m"] == 0
    assert row["qty_ratio_3m"] == 0


def test_no_previous_orders_do_not_invent_a_ratio():
    keys, headers, lines = example_history()
    headers = headers[headers.SalesOrderID.eq(4)]
    lines = lines[lines.SalesOrderID.eq(4)]
    row = build_features(keys, headers, lines).iloc[0]
    assert np.isnan(row["qty_ratio_3m"])
    assert np.isnan(row["discount_change_3m"])


def test_protected_snapshot_is_rejected():
    keys, headers, lines = example_history()
    keys["snapshot"] = pd.Timestamp("2013-10-01")
    with pytest.raises(ValueError, match="approved development"):
        build_features(keys, headers, lines)


def test_duplicate_order_line_is_rejected_before_aggregation():
    _, headers, lines = example_history()
    lines.loc[1, "SalesOrderDetailID"] = lines.loc[0, "SalesOrderDetailID"]
    with pytest.raises(ValueError, match="Duplicate order lines"):
        clean_extracts(headers, lines, headers)


def test_existing_product_is_not_counted_as_new():
    keys, headers, lines = example_history()
    lines.loc[lines.SalesOrderID.eq(4), "ProductID"] = 1
    row = build_features(keys, headers, lines).iloc[0]
    assert row["new_product_share_3m"] == 0


def test_accepted_order_without_details_is_rejected():
    _, headers, lines = example_history()
    lines = lines[lines.SalesOrderID.ne(4)]
    with pytest.raises(ValueError, match="no detail lines"):
        clean_extracts(headers, lines, headers)

"""Exact collision signatures discard IDs, preserve events, and exclude future."""

import pandas as pd

from comparisons.history_collisions import collision_tables, history_signatures


def test_history_signature_ignores_ids_but_keeps_product_and_past_boundary():
    keys = pd.DataFrame({"StoreID": [1, 2], "snapshot": pd.Timestamp("2013-04-01")})
    orders = pd.DataFrame(
        {
            "StoreID": [1, 2, 1],
            "SalesOrderID": [10, 20, 30],
            "OrderDate": pd.to_datetime(["2013-03-31", "2013-03-31", "2013-04-01"]),
            "SubTotal": [10.0, 10.0, 99.0],
        }
    )
    lines = pd.DataFrame(
        {
            "SalesOrderID": [10, 20, 30],
            "ProductID": [7, 7, 9],
            "OrderQty": [1, 1, 1],
            "UnitPrice": [10.0, 10.0, 99.0],
            "UnitPriceDiscount": 0.0,
            "LineTotal": [10.0, 10.0, 99.0],
        }
    )
    result = history_signatures(keys, orders, lines)
    assert result.groupby("representation").signature.nunique().eq(1).all()
    lines.loc[1, "ProductID"] = 8
    changed = history_signatures(keys, orders, lines)
    assert changed.groupby("representation").signature.nunique().eq(2).all()


def test_collisions_do_not_cross_snapshots_and_report_mixed_cells():
    cases = pd.DataFrame(
        {
            "StoreID": [1, 2, 3],
            "snapshot": pd.to_datetime(["2013-01-01", "2013-01-01", "2013-04-01"]),
            "representation": "example",
            "signature": "same",
            "churn": [0, 1, 0],
        }
    )
    annotated, summary = collision_tables(cases)
    assert annotated.cell_rows.tolist() == [2, 2, 1]
    assert annotated.mixed_outcome.tolist() == [True, True, False]
    assert summary.mixed_rows.sum() == 2

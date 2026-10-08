import pandas as pd

from comparisons.correlation_pruned_churn.corrected_features import HistoricalSnapshotBuilder
from comparisons.correlation_pruned_churn.corrected_model import select_numeric_features


def sample_orders() -> pd.DataFrame:
    rows = [
        (1, 101, "2013-07-01", 100, 1, 1, 0, 0, 100, 5, 270),
        (2, 102, "2013-07-01", 200, 2, 1, 0, 0, 200, 6, 275),
        (2, 103, "2013-12-01", 300, 1, 1, 0, 0, 300, 6, 275),
        # May 1 is OUTSIDE the six months starting November 1.
        (1, 104, "2014-05-01", 500, 5, 1, 0, 0, 500, 5, 270),
    ]
    columns = [
        "StoreID", "SalesOrderID", "OrderDate", "SubTotal",
        "qty", "n_lines", "avg_disc", "rev_components",
        "rev_bikes", "TerritoryID", "SalesPersonID",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result["OrderDate"] = pd.to_datetime(result["OrderDate"])
    result["rev_clothing"] = 0
    result["rev_accessories"] = 0
    return result


def test_corrected_six_month_label_excludes_end_date() -> None:
    stores = pd.DataFrame(
        {"StoreID": [1, 2], "YearOpened": [2000, 2001]}
    )
    builder = HistoricalSnapshotBuilder(sample_orders(), stores)

    frame = builder.build(("2013-11-01",), "full")
    labels = dict(zip(frame["StoreID"], frame["churn"]))

    assert labels == {1: 1, 2: 0}


def test_future_orders_never_change_snapshot_features() -> None:
    stores = pd.DataFrame(
        {"StoreID": [1, 2], "YearOpened": [2000, 2001]}
    )
    orders = sample_orders()
    builder = HistoricalSnapshotBuilder(orders, stores)

    before = builder.build(("2013-11-01",), "full")

    new_order = orders.loc[orders["SalesOrderID"] == 104].copy()
    new_order["OrderDate"] = pd.Timestamp("2013-12-15")
    new_order["SalesOrderID"] = 105

    revised = HistoricalSnapshotBuilder(
        pd.concat([orders, new_order], ignore_index=True),
        stores,
    ).build(("2013-11-01",), "full")

    # Future transactions are allowed to change only the churn label.
    columns = [col for col in before if col != "churn"]
    pd.testing.assert_frame_equal(
        before[columns].reset_index(drop=True),
        revised[columns].reset_index(drop=True),
    )


def test_feature_selection_uses_supplied_training_rows_only() -> None:
    train = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [0, 1, 2, 3, 4, 5],
            "churn": [0, 0, 0, 1, 1, 1],
        }
    )
    selected = select_numeric_features(train, ["x", "y"])

    # This function receives training rows only; validation labels
    # are neither a parameter nor a global input.
    assert len(selected) == 1
    assert selected[0] in ("x", "y")

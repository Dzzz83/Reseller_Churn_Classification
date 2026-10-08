"""Independently check selected feature calculations against raw orders."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from comparisons.correlation_pruned_churn.corrected_features import (
    FriendSnapshotBuilder,
)
from comparisons.correlation_pruned_churn.validation import DEVELOPMENT_SNAPSHOTS
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


SHARE_COLUMNS = (
    "share_bikes",
    "share_components",
    "share_clothing",
    "share_accessories",
)


def verify_sample(
    frame: pd.DataFrame,
    orders: pd.DataFrame,
    variant: str,
) -> int:
    comparisons = 0
    for snapshot, group in frame.groupby("snapshot"):
        end = pd.Timestamp(snapshot)
        for row in group.head(10).itertuples(index=False):
            store_orders = orders[
                (orders["StoreID"] == row.StoreID)
                & (orders["OrderDate"] < end)
            ]
            if variant == "obs6":
                store_orders = store_orders[
                    store_orders["OrderDate"]
                    >= end - pd.DateOffset(months=6)
                ]

            if len(store_orders) == 0:
                raise AssertionError("Eligible reseller lacks history")

            expected_recency = (
                end - store_orders["OrderDate"].max()
            ).days
            expected_orders = store_orders[
                "SalesOrderID"
            ].nunique()
            expected_revenue = store_orders[
                "SubTotal"
            ].sum()
            if row.recency_days != expected_recency:
                raise AssertionError("Incorrect recency")
            if row.n_orders_total != expected_orders:
                raise AssertionError("Incorrect historical order count")
            if not np.isclose(
                row.revenue_total, expected_revenue, rtol=1e-8
            ):
                raise AssertionError("Incorrect historical revenue")

            recent_window = 3 if variant == "obs6" else 6
            recent = store_orders[
                store_orders["OrderDate"]
                >= end - pd.DateOffset(months=recent_window)
            ]
            expected_recent_orders = recent["SalesOrderID"].nunique()
            expected_recent_revenue = recent["SubTotal"].sum()
            if getattr(row, f"n_orders_{recent_window}m") != expected_recent_orders:
                raise AssertionError("Incorrect recent order count")
            if not np.isclose(
                getattr(row, f"revenue_{recent_window}m"),
                expected_recent_revenue,
                rtol=1e-8,
                atol=1e-6,
            ):
                raise AssertionError("Incorrect recent revenue")
            comparisons += 1

    return comparisons


def main() -> None:
    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()
    builder = FriendSnapshotBuilder(orders, stores)

    print("=== Friend Feature Formula Verification ===")
    for variant in ("full", "obs6"):
        frame = builder.build(
            DEVELOPMENT_SNAPSHOTS,
            feature_window=variant,
        )

        if (frame["revenue_total"] <= 0).any():
            raise AssertionError("Nonpositive historical revenue")
        shares = frame[list(SHARE_COLUMNS)]
        if ((shares < -1e-6) | (shares > 1 + 1e-6)).any().any():
            raise AssertionError("Product share outside [0, 1]")
        if not np.allclose(
            shares.sum(axis=1),
            1,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError("Product shares do not sum to 1")

        if frame["last_order_salesperson"].isna().any():
            raise AssertionError("Historical salesperson is missing")
        if frame["TerritoryID"].isna().any():
            raise AssertionError("Historical order territory is missing")

        checked = verify_sample(
            frame=frame,
            orders=orders,
            variant=variant,
        )

        print(
            f"[PASS] {variant}: {len(frame)} rows; "
            f"{checked} independently recalculated examples"
        )
        print(
            "[PASS] product shares, historical salesperson, "
            "order territory"
        )

    print("Final-test labels NOT accessed.")


if __name__ == "__main__":
    main()

"""Verify corrected source data and historical churn labels.

This audit intentionally constructs only development snapshots ending
before the protected 2013-10-01 final test date.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from comparisons.correlation_pruned_churn.corrected_features import HistoricalSnapshotBuilder
from comparisons.correlation_pruned_churn.validation import (
    DEVELOPMENT_SNAPSHOTS,
    ValidationPlans,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


OUTPUT = ROOT / "results" / "correlation_pruned_churn"


def main() -> None:
    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    if len(orders) != 3800 or len(stores) != 701:
        raise AssertionError(
            "Expected 3,800 valid orders and 701 store records"
        )
    if orders["SalesOrderID"].duplicated().any():
        raise AssertionError("Duplicate SalesOrderID")
    if stores["StoreID"].duplicated().any():
        raise AssertionError("Duplicate StoreID")
    if orders["StoreID"].isna().any():
        raise AssertionError("Missing order StoreID")
    if orders["OrderDate"].isna().any():
        raise AssertionError("Invalid order date")
    if (
        (orders["SubTotal"] <= 0).any()
        or orders["SubTotal"].isna().any()
        or (orders["qty"] <= 0).any()
        or orders["qty"].isna().any()
    ):
        raise AssertionError("Invalid order values")
    if not set(orders["StoreID"]).issubset(set(stores["StoreID"])):
        raise AssertionError("Unmatched reseller in order table")

    revenue_columns = [
        "rev_bikes",
        "rev_components",
        "rev_clothing",
        "rev_accessories",
    ]
    if not np.allclose(
        orders[revenue_columns].sum(axis=1),
        orders["SubTotal"],
        rtol=1e-6,
        atol=0.02,
    ):
        raise AssertionError("Category revenue does not reconcile")

    builder = HistoricalSnapshotBuilder(orders=orders, stores=stores)
    features = {
        variant: builder.build(
            DEVELOPMENT_SNAPSHOTS,
            feature_window=variant,
        )
        for variant in ("full", "obs6")
    }

    keys = ["snapshot", "StoreID", "churn"]
    full_keys = (
        features["full"][keys]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    short_keys = (
        features["obs6"][keys]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(full_keys, short_keys)

    for variant, frame in features.items():
        ValidationPlans.verify_development_period(frame)
        if frame["churn"].isna().any():
            raise AssertionError(f"{variant}: missing churn labels")
        if not frame["churn"].isin((0, 1)).all():
            raise AssertionError(f"{variant}: invalid churn labels")

    summary = (
        features["full"]
        .groupby("snapshot")
        .agg(
            eligible=("StoreID", "size"),
            churners=("churn", "sum"),
        )
        .reset_index()
    )
    summary["churn_rate"] = (
        summary["churners"] / summary["eligible"]
    )
    expected_counts = {
        "2012-05-01": (200, 70),
        "2012-08-01": (393, 74),
        "2012-11-01": (347, 28),
        "2013-02-01": (343, 29),
    }
    actual_counts = {
        row.snapshot: (int(row.eligible), int(row.churners))
        for row in summary.itertuples(index=False)
    }
    if actual_counts != expected_counts:
        raise AssertionError(
            "Corrected snapshot/label counts differ from the "
            f"verified cleaned-source reference: {actual_counts}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "01_verified_snapshot_counts.csv"
    summary.to_csv(path, index=False)

    print("=== Source Data and Churn Labels ===")
    print("[PASS] 3,800 cleaned orders and 701 stores")
    print("[PASS] IDs, values, dates, reseller joins, revenue")
    print("[PASS] Two feature variants have identical StoreID/snapshot/label")
    print("[PASS] No development label extends past 2013-10-01")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("Final-test labels NOT accessed.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

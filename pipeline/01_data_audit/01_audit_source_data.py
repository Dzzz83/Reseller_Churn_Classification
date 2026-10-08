from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from reseller_churn.data.dataset_loader import DatasetLoader


REQUIRED_ORDER_COLUMNS = {
    "StoreID",
    "SalesOrderID",
    "OrderDate",
    "SubTotal",
    "qty",
    "rev_bikes",
    "rev_components",
    "rev_clothing",
    "rev_accessories",
}


def main() -> None:
    print("=== 01. Data Audit ===")

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    missing_columns = REQUIRED_ORDER_COLUMNS - set(orders.columns)
    assert not missing_columns, f"Missing order columns: {missing_columns}"

    assert orders["SalesOrderID"].is_unique
    assert not orders.duplicated().any()
    assert orders["OrderDate"].notna().all()
    assert (orders["SubTotal"] > 0).all()
    assert orders["qty"].notna().all()
    assert (orders["qty"] > 0).all()
    assert orders["StoreID"].notna().all()

    assert stores["StoreID"].is_unique
    assert stores["StoreID"].notna().all()

    unknown_store_ids = (
        set(orders["StoreID"])
        - set(stores["StoreID"])
    )
    assert not unknown_store_ids

    reconstructed_revenue = (
        orders["rev_bikes"]
        + orders["rev_components"]
        + orders["rev_clothing"]
        + orders["rev_accessories"]
    )

    assert np.allclose(
        reconstructed_revenue,
        orders["SubTotal"],
        atol=0.01,
        rtol=0,
    )

    print(f"[PASS] Orders: {len(orders):,} rows")
    print(f"[PASS] Stores: {len(stores):,} rows")
    print(
        "[PASS] IDs, dates, quantities, revenue, store joins, "
        "and category revenue are valid."
    )


if __name__ == "__main__":
    main()

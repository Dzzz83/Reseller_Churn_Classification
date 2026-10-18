"""Validation shared by the preparation runner and historical source audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reseller_churn.data.validation import validate_category_revenue, validate_order_dates


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


def audit_source_data(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
) -> dict[str, object]:
    """Apply the existing Stage 01 contract and summarize its checks."""
    missing_columns = REQUIRED_ORDER_COLUMNS - set(orders.columns)
    missing_store_columns = {"StoreID"} - set(stores.columns)
    if missing_columns or missing_store_columns:
        raise ValueError(
            "Missing source columns: "
            f"orders={sorted(missing_columns)}, "
            f"stores={sorted(missing_store_columns)}"
        )

    validate_order_dates(orders.OrderDate)
    validate_category_revenue(orders)
    quantities = orders.qty.to_numpy(dtype=float)
    if not np.isfinite(quantities).all() or (quantities <= 0).any():
        raise ValueError("Order quantities must be finite and positive.")
    duplicate_order_ids = int(orders["SalesOrderID"].duplicated().sum())
    duplicate_order_rows = int(orders.duplicated().sum())
    missing_order_dates = int(orders["OrderDate"].isna().sum())
    nonpositive_subtotals = int((orders["SubTotal"] <= 0).sum())
    missing_quantities = int(orders["qty"].isna().sum())
    nonpositive_quantities = int((orders["qty"].fillna(0) <= 0).sum())
    missing_order_store_ids = int(orders["StoreID"].isna().sum())
    duplicate_store_ids = int(stores["StoreID"].duplicated().sum())
    missing_store_ids = int(stores["StoreID"].isna().sum())
    unknown_store_ids = len(set(orders["StoreID"]) - set(stores["StoreID"]))
    reconstructed_revenue = orders[
        ["rev_bikes", "rev_components", "rev_clothing", "rev_accessories"]
    ].sum(axis=1, min_count=4)
    revenue_mismatches = int(
        (~np.isclose(
            reconstructed_revenue,
            orders["SubTotal"],
            atol=0.01,
            rtol=0,
        )).sum()
    )

    checks = {
        "required_source_columns_present": (
            not missing_columns and not missing_store_columns
        ),
        "sales_order_id_unique": duplicate_order_ids == 0,
        "order_rows_unique": duplicate_order_rows == 0,
        "order_dates_present": missing_order_dates == 0,
        "subtotals_positive": nonpositive_subtotals == 0,
        "quantities_present_and_positive": (
            missing_quantities == 0 and nonpositive_quantities == 0
        ),
        "order_store_ids_present": missing_order_store_ids == 0,
        "store_ids_unique_and_present": (
            duplicate_store_ids == 0 and missing_store_ids == 0
        ),
        "all_order_stores_known": unknown_store_ids == 0,
        "category_revenue_matches_subtotal": revenue_mismatches == 0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"Source data audit failed: {failed_checks}")

    return {
        "order_rows": len(orders),
        "store_rows": len(stores),
        "required_order_columns": sorted(REQUIRED_ORDER_COLUMNS),
        "required_store_columns": ["StoreID"],
        "order_columns": list(orders.columns),
        "store_columns": list(stores.columns),
        "missing_order_dates": missing_order_dates,
        "missing_order_store_ids": missing_order_store_ids,
        "duplicate_order_ids": duplicate_order_ids,
        "duplicate_order_rows": duplicate_order_rows,
        "duplicate_store_ids": duplicate_store_ids,
        "missing_store_ids": missing_store_ids,
        "unknown_store_ids": unknown_store_ids,
        "nonpositive_subtotals": nonpositive_subtotals,
        "missing_quantities": missing_quantities,
        "nonpositive_quantities": nonpositive_quantities,
        "category_revenue_mismatches": revenue_mismatches,
        "checks": checks,
    }

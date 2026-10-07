from pathlib import Path
import sys

import pandas as pd


DATA_DIR = Path("datasets")
ORDERS_FILE = DATA_DIR / "orders_clean.csv"
STORES_FILE = DATA_DIR / "stores_clean.csv"


def check(name, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")

    if details:
        print(f"       {details}")

    return condition


def main():
    print("\n=== Stage 1A: Input Dataset Audit ===\n")

    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------
    orders = pd.read_csv(ORDERS_FILE)
    stores = pd.read_csv(STORES_FILE)

    orders["OrderDate"] = pd.to_datetime(
        orders["OrderDate"],
        errors="coerce"
    )

    print("Dataset summary")
    print("----------------")
    print(f"Orders rows:  {len(orders):,}")
    print(f"Stores rows:  {len(stores):,}")
    print(f"Order period: {orders['OrderDate'].min().date()} "
          f"-> {orders['OrderDate'].max().date()}")
    print(f"Order resellers: {orders['StoreID'].nunique():,}")
    print(f"Store records:   {stores['StoreID'].nunique():,}")
    print()

    passed = []

    # --------------------------------------------------
    # 2. Required columns
    # --------------------------------------------------
    required_order_columns = {
        "StoreID",
        "SalesOrderID",
        "OrderDate",
        "TerritoryID",
        "SalesPersonID",
        "SubTotal",
        "qty",
        "n_lines",
        "avg_disc",
        "rev_bikes",
        "rev_components",
        "rev_clothing",
        "rev_accessories",
    }

    required_store_columns = {
        "StoreID",
        "SalesPersonID",
        "AnnualSales",
        "AnnualRevenue",
        "BusinessType",
        "YearOpened",
        "Specialty",
        "SquareFeet",
        "Brands",
        "Internet",
        "NumberEmployees",
    }

    passed.append(
        check(
            "Orders contains required columns",
            required_order_columns.issubset(orders.columns),
        )
    )

    passed.append(
        check(
            "Stores contains required columns",
            required_store_columns.issubset(stores.columns),
        )
    )

    # --------------------------------------------------
    # 3. Order integrity
    # --------------------------------------------------
    passed.append(
        check(
            "SalesOrderID is unique",
            not orders["SalesOrderID"].duplicated().any(),
            f"Duplicates: {orders['SalesOrderID'].duplicated().sum()}",
        )
    )

    passed.append(
        check(
            "No duplicated order rows",
            not orders.duplicated().any(),
            f"Duplicates: {orders.duplicated().sum()}",
        )
    )

    passed.append(
        check(
            "All OrderDate values are valid",
            orders["OrderDate"].notna().all(),
            f"Invalid dates: {orders['OrderDate'].isna().sum()}",
        )
    )

    passed.append(
        check(
            "All SubTotal values are positive",
            (orders["SubTotal"] > 0).all(),
            f"Invalid rows: {(orders['SubTotal'] <= 0).sum()}",
        )
    )

    passed.append(
        check(
            "All qty values are positive",
            (orders["qty"] > 0).all(),
            f"Invalid rows: {(orders['qty'] <= 0).sum()}",
        )
    )

    passed.append(
        check(
            "No missing StoreID in orders",
            orders["StoreID"].notna().all(),
            f"Missing: {orders['StoreID'].isna().sum()}",
        )
    )

    # --------------------------------------------------
    # 4. Store integrity
    # --------------------------------------------------
    passed.append(
        check(
            "StoreID is unique in stores",
            not stores["StoreID"].duplicated().any(),
            f"Duplicates: {stores['StoreID'].duplicated().sum()}",
        )
    )

    passed.append(
        check(
            "No missing StoreID in stores",
            stores["StoreID"].notna().all(),
            f"Missing: {stores['StoreID'].isna().sum()}",
        )
    )

    # --------------------------------------------------
    # 5. Orders -> Stores relationship
    # --------------------------------------------------
    order_store_ids = set(orders["StoreID"])
    store_ids = set(stores["StoreID"])

    missing_store_ids = order_store_ids - store_ids

    passed.append(
        check(
            "Every order reseller exists in stores",
            len(missing_store_ids) == 0,
            f"Missing StoreIDs: {sorted(missing_store_ids)}",
        )
    )

    # --------------------------------------------------
    # 6. Missing-value report
    # --------------------------------------------------
    print("\nMissing values")
    print("----------------")

    order_missing = orders.isna().sum()
    order_missing = order_missing[order_missing > 0]

    store_missing = stores.isna().sum()
    store_missing = store_missing[store_missing > 0]

    if order_missing.empty:
        print("Orders: none")
    else:
        print("\nOrders:")
        print(order_missing.to_string())

    if store_missing.empty:
        print("Stores: none")
    else:
        print("\nStores:")
        print(store_missing.to_string())

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------
    print("\n=== Audit Result ===")

    if all(passed):
        print("PASS: datasets are ready for Stage 1B.")
        sys.exit(0)

    print("FAIL: fix the issues above before Stage 1B.")
    sys.exit(1)


if __name__ == "__main__":
    main()
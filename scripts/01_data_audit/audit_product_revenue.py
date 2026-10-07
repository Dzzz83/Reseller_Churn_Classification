
import numpy as np
import pandas as pd


DATA_PATH = "datasets/orders_clean.csv"

REVENUE_COLUMNS = [
    "rev_bikes",
    "rev_components",
    "rev_clothing",
    "rev_accessories",
]


def main():
    orders = pd.read_csv(DATA_PATH)

    print("=== Stage 1E.7: Product Revenue Audit ===\n")

    # Calculate total revenue across product categories.
    category_total = orders[REVENUE_COLUMNS].sum(axis=1)

    difference = (
        orders["SubTotal"] - category_total
    ).abs()

    # Check for missing or negative category revenue.
    missing = orders[REVENUE_COLUMNS].isna().sum()
    negative = (orders[REVENUE_COLUMNS] < 0).sum()

    print("Missing values:")
    print(missing.to_string())

    print("\nNegative values:")
    print(negative.to_string())

    print("\nRevenue reconciliation:")
    print(f"Orders checked: {len(orders)}")
    print(f"Maximum difference: {difference.max():.6f}")

    mismatches = ~np.isclose(
        orders["SubTotal"],
        category_total,
        rtol=0,
        atol=0.01,
    )

    print(f"Mismatched orders: {mismatches.sum()}")

    if mismatches.any():
        print("\nSample mismatches:")
        print(
            orders.loc[
                mismatches,
                ["SalesOrderID", "SubTotal"] + REVENUE_COLUMNS
            ].head(5).to_string(index=False)
        )

    # Final checks.
    assert missing.sum() == 0
    assert negative.sum() == 0
    assert mismatches.sum() == 0

    print("\n[PASS] Product revenue data is consistent")


if __name__ == "__main__":
    main()

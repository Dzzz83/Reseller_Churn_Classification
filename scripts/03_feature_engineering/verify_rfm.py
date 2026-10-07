
import pandas as pd


ORDERS_PATH = "datasets/orders_clean.csv"
FEATURES_PATH = "datasets/processed/rfm_features.csv"

SNAPSHOT = pd.Timestamp("2012-07-01")


def verify_reseller(orders, row, name):
    store_id = int(row["StoreID"])
    history_start = SNAPSHOT - pd.DateOffset(months=12)

    # Retrieve original orders for this reseller
    history = orders[
        (orders["StoreID"] == store_id)
        & (orders["OrderDate"] >= history_start)
        & (orders["OrderDate"] < SNAPSHOT)
    ].sort_values("OrderDate")

    # Recalculate directly from source transactions
    expected_recency = (
        SNAPSHOT - history["OrderDate"].max()
    ).days

    expected_frequency = history["SalesOrderID"].nunique()
    expected_revenue = sum(history["SubTotal"])

    print(f"\n{'=' * 60}")
    print(f"{name} | StoreID: {store_id}")
    print("=" * 60)

    print("\nOriginal orders:")
    print(
        history[
            ["SalesOrderID", "OrderDate", "SubTotal"]
        ].to_string(index=False)
    )

    print("\nFeature comparison:")
    print(f"{'Feature':<20} {'Expected':>15} {'Generated':>15}")

    comparisons = {
        "recency_days": (
            expected_recency,
            row["recency_days"],
        ),
        "n_orders_12m": (
            expected_frequency,
            row["n_orders_12m"],
        ),
        "revenue_12m": (
            expected_revenue,
            row["revenue_12m"],
        ),
    }

    for feature, (expected, generated) in comparisons.items():
        print(
            f"{feature:<20} "
            f"{expected:>15.2f} "
            f"{generated:>15.2f}"
        )

        tolerance = 0.01 if feature == "revenue_12m" else 0
        assert abs(expected - generated) <= tolerance

    # Confirm eligibility and historical-only data
    assert not history.empty
    assert history["OrderDate"].max() < SNAPSHOT
    assert (
        history["OrderDate"]
        >= SNAPSHOT - pd.DateOffset(months=6)
    ).any()

    print("\n[PASS] All calculations match source orders")


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
    )

    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    snapshot_features = features[
        features["snapshot"] == SNAPSHOT
    ]

    assert len(snapshot_features) == 326

    # Highest revenue reseller
    highest = snapshot_features.loc[
        snapshot_features["revenue_12m"].idxmax()
    ]

    remaining = snapshot_features[
        snapshot_features["StoreID"] != highest["StoreID"]
    ]

    # One-order reseller
    one_order = remaining[
        remaining["n_orders_12m"] == 1
    ].sort_values("StoreID").iloc[0]

    # Multiple-order reseller
    multiple = remaining[
        (remaining["n_orders_12m"] >= 2)
        & (remaining["StoreID"] != one_order["StoreID"])
    ].sort_values("StoreID").iloc[0]

    print("=== Stage 1E.3: RFM Verification ===")

    examples = [
        ("One-order reseller", one_order),
        ("Multiple-order reseller", multiple),
        ("Highest-revenue reseller", highest),
    ]

    for name, row in examples:
        verify_reseller(orders, row, name)

    print("\n[PASS] All 3 reseller examples verified")


if __name__ == "__main__":
    main()

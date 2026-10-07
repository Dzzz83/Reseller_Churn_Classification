
from pathlib import Path

import pandas as pd

from compare_split_strategies import make_window
from define_windows import select_period


DATA_PATH = Path("datasets/orders_clean.csv")

SNAPSHOTS = [
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
    "2013-07-01",
    "2013-10-01",
]


def build_rfm(orders, snapshot):
    """Build RFM features using only historical orders."""
    snapshot = pd.Timestamp(snapshot)
    window = make_window(snapshot)

    # Full 12-month observation history
    history = select_period(
        orders,
        window["observation_start"],
        snapshot,
    )

    # Eligibility: at least one order in last 6 months
    recent_start = snapshot - pd.DateOffset(months=6)

    eligible_ids = history.loc[
        history["OrderDate"] >= recent_start,
        "StoreID",
    ].unique()

    eligible_history = history[
        history["StoreID"].isin(eligible_ids)
    ]

    # Aggregate historical purchasing behavior
    features = (
        eligible_history.groupby("StoreID")
        .agg(
            last_order=("OrderDate", "max"),
            n_orders_12m=("SalesOrderID", "nunique"),
            revenue_12m=("SubTotal", "sum"),
        )
        .reset_index()
    )

    features["recency_days"] = (
        snapshot - features["last_order"]
    ).dt.days

    features["snapshot"] = snapshot

    # Keep last_order internally for auditing only
    return features[
        [
            "StoreID",
            "snapshot",
            "last_order",
            "recency_days",
            "n_orders_12m",
            "revenue_12m",
        ]
    ]


def verify_rfm(features, orders, snapshot):
    """Verify RFM calculations against source orders."""
    snapshot = pd.Timestamp(snapshot)
    window = make_window(snapshot)

    history = select_period(
        orders,
        window["observation_start"],
        snapshot,
    )

    assert features["StoreID"].is_unique
    assert (features["snapshot"] == snapshot).all()

    # Every eligible reseller has at least one order
    assert (features["n_orders_12m"] >= 1).all()
    assert (features["revenue_12m"] > 0).all()

    # Historical-only features
    assert (features["last_order"] < snapshot).all()
    assert (features["recency_days"] >= 1).all()
    assert (features["recency_days"] <= 184).all()

    # Independently calculate expected eligibility
    recent_start = snapshot - pd.DateOffset(months=6)
    expected_ids = set(
        history.loc[
            history["OrderDate"] >= recent_start,
            "StoreID",
        ]
    )

    assert set(features["StoreID"]) == expected_ids

    # Verify all resellers, not just a sample
    expected = (
        history[history["StoreID"].isin(expected_ids)]
        .groupby("StoreID")
        .agg(
            expected_last_order=("OrderDate", "max"),
            expected_orders=("SalesOrderID", "nunique"),
            expected_revenue=("SubTotal", "sum"),
        )
    )

    actual = features.set_index("StoreID")

    assert actual["last_order"].equals(
        expected["expected_last_order"]
    )

    assert (
        actual["n_orders_12m"]
        == expected["expected_orders"]
    ).all()

    assert (
        (
            actual["revenue_12m"]
            - expected["expected_revenue"]
        ).abs() < 0.01
    ).all()

    print(f"[PASS] {snapshot.date()}: {len(features)} resellers")


def main():
    orders = pd.read_csv(
        DATA_PATH,
        parse_dates=["OrderDate"],
    )

    print("=== Stage 1E.2: Basic RFM Features ===\n")

    all_features = []

    for snapshot in SNAPSHOTS:
        features = build_rfm(orders, snapshot)

        verify_rfm(
            features,
            orders,
            snapshot,
        )

        all_features.append(features)

    combined = pd.concat(
        all_features,
        ignore_index=True,
    )

    assert not combined.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    print("\nFeature summary")
    print("-" * 50)

    print(
        combined[
            ["recency_days", "n_orders_12m", "revenue_12m"]
        ].describe().round(2)
    )

    print("\nMissing values")
    print(
        combined[
            ["recency_days", "n_orders_12m", "revenue_12m"]
        ].isna().sum()
    )

    # Save features for later pipeline stages
    output_dir = Path("datasets/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "rfm_features.csv"

    # Do not export the audit-only last_order column
    combined.drop(columns="last_order").to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")
    print(f"Total observations: {len(combined)}")
    print("\n[PASS] RFM feature generation completed")


if __name__ == "__main__":
    main()

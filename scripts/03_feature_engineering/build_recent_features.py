
from pathlib import Path

import pandas as pd


ORDERS_PATH = Path("datasets/orders_clean.csv")
RFM_PATH = Path("datasets/processed/rfm_features.csv")
OUTPUT_PATH = Path("datasets/processed/rfm_recent_features.csv")


def build_recent_features(orders, rfm):
    results = []

    for snapshot, base in rfm.groupby("snapshot"):
        base = base.copy()

        for months in (6, 3):
            start = snapshot - pd.DateOffset(months=months)

            history = orders[
                (orders["OrderDate"] >= start)
                & (orders["OrderDate"] < snapshot)
            ]

            features = (
                history.groupby("StoreID")
                .agg(
                    **{
                        f"n_orders_{months}m": (
                            "SalesOrderID", "nunique"
                        ),
                        f"revenue_{months}m": (
                            "SubTotal", "sum"
                        ),
                    }
                )
                .reset_index()
            )

            base = base.merge(
                features,
                on="StoreID",
                how="left",
                validate="one_to_one",
            )

            columns = [
                f"n_orders_{months}m",
                f"revenue_{months}m",
            ]

            base[columns] = base[columns].fillna(0)
            base[f"n_orders_{months}m"] = (
                base[f"n_orders_{months}m"].astype(int)
            )

        results.append(base)

    return pd.concat(results, ignore_index=True)


def verify(features, rfm):
    # Original observations must remain unchanged
    assert len(features) == len(rfm)
    assert not features.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    # Recent activity cannot exceed longer-term activity
    assert (
        features["n_orders_3m"]
        <= features["n_orders_6m"]
    ).all()

    assert (
        features["n_orders_6m"]
        <= features["n_orders_12m"]
    ).all()

    tolerance = 0.01

    assert (
        features["revenue_3m"]
        <= features["revenue_6m"] + tolerance
    ).all()

    assert (
        features["revenue_6m"]
        <= features["revenue_12m"] + tolerance
    ).all()

    # Eligibility guarantees recent 6-month activity
    assert (features["n_orders_6m"] >= 1).all()
    assert (features["revenue_6m"] > 0).all()

    # Three-month inactivity is allowed
    recent_columns = [
        "n_orders_3m",
        "n_orders_6m",
        "revenue_3m",
        "revenue_6m",
    ]

    assert features[recent_columns].notna().all().all()
    assert (features[recent_columns] >= 0).all().all()

    # Manual spot checks based on verified source orders
    july = features[
        features["snapshot"] == pd.Timestamp("2012-07-01")
    ].set_index("StoreID")

    assert july.loc[292, "n_orders_6m"] == 2
    assert july.loc[292, "n_orders_3m"] == 1

    assert july.loc[616, "n_orders_6m"] == 2
    assert july.loc[616, "n_orders_3m"] == 0

    print("[PASS] Observation counts unchanged")
    print("[PASS] Recent activity constraints")
    print("[PASS] Eligibility consistency")
    print("[PASS] Missing-value checks")
    print("[PASS] Manual reseller checks")


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
    )

    rfm = pd.read_csv(
        RFM_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 1E.4: Recent Purchasing Features ===\n")

    features = build_recent_features(orders, rfm)

    verify(features, rfm)

    print("\nFeature summary")
    print("-" * 55)

    columns = [
        "n_orders_6m",
        "n_orders_3m",
        "revenue_6m",
        "revenue_3m",
    ]

    print(features[columns].describe().round(2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total observations: {len(features)}")
    print("\n[PASS] Stage 1E.4 completed")


if __name__ == "__main__":
    main()

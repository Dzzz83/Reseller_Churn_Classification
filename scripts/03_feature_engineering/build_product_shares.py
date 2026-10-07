
from pathlib import Path

import numpy as np
import pandas as pd


ORDERS_PATH = Path("datasets/orders_clean.csv")
INPUT_PATH = Path(
    "datasets/processed/rfm_recent_gaps_trend_features.csv"
)
OUTPUT_PATH = Path(
    "datasets/processed/reseller_order_features.csv"
)

CATEGORIES = [
    "bikes",
    "components",
    "clothing",
    "accessories",
]

TOLERANCE = 1e-4


def build_product_shares(orders, base):
    results = []

    for snapshot, group in base.groupby("snapshot"):
        start = snapshot - pd.DateOffset(months=12)

        # Use only the previous 12 months.
        history = orders[
            (orders["OrderDate"] >= start)
            & (orders["OrderDate"] < snapshot)
        ]

        revenue_columns = [
            f"rev_{category}" for category in CATEGORIES
        ]

        # Aggregate revenue by reseller.
        aggregated = (
            history.groupby("StoreID")[revenue_columns]
            .sum()
            .reset_index()
        )

        result = group.merge(
            aggregated,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        # Calculate revenue proportions.
        for category in CATEGORIES:
            result[f"share_{category}"] = (
                result[f"rev_{category}"]
                / result["revenue_12m"]
            )

        # Remove intermediate revenue columns.
        result = result.drop(columns=revenue_columns)

        results.append(result)

    return pd.concat(results, ignore_index=True)


def verify(features, base, orders):
    # Original observations and features must remain unchanged.
    assert len(features) == len(base) == 2312

    pd.testing.assert_frame_equal(
        features[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
    )

    assert not features.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    print("[PASS] Original features preserved")

    share_columns = [
        f"share_{category}" for category in CATEGORIES
    ]

    shares = features[share_columns]

    # No missing or infinite values.
    assert shares.notna().all().all()
    assert np.isfinite(shares.to_numpy()).all()

    # Each share must be between 0 and 1.
    assert (shares >= -TOLERANCE).all().all()
    assert (shares <= 1 + TOLERANCE).all().all()

    print("[PASS] Product share ranges")

    # All four shares should total approximately 100%.
    share_totals = shares.sum(axis=1)

    assert np.allclose(
        share_totals,
        1.0,
        rtol=0,
        atol=TOLERANCE,
    )

    print("[PASS] Product shares sum to 100%")

    # Independently verify three previously audited resellers.
    snapshot = pd.Timestamp("2012-07-01")
    start = snapshot - pd.DateOffset(months=12)

    for store_id in (326, 292, 616):
        history = orders[
            (orders["StoreID"] == store_id)
            & (orders["OrderDate"] >= start)
            & (orders["OrderDate"] < snapshot)
        ]

        row = features[
            (features["StoreID"] == store_id)
            & (features["snapshot"] == snapshot)
        ].iloc[0]

        total = history["SubTotal"].sum()

        for category in CATEGORIES:
            expected = (
                history[f"rev_{category}"].sum() / total
            )

            actual = row[f"share_{category}"]

            assert np.isclose(
                actual,
                expected,
                rtol=0,
                atol=1e-8,
            )

    print("[PASS] Three reseller calculations")


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
    )

    base = pd.read_csv(
        INPUT_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 1E.7: Product Share Features ===\n")

    features = build_product_shares(orders, base)

    verify(features, base, orders)

    share_columns = [
        f"share_{category}" for category in CATEGORIES
    ]

    print("\nProduct share summary:")
    print(
        features[share_columns]
        .describe()
        .round(3)
    )

    print("\nMissing values:")
    print(features[share_columns].isna().sum())

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total observations: {len(features)}")

    print("\n[PASS] Product share generation completed")


if __name__ == "__main__":
    main()

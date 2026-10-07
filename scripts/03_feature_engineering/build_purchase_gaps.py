
from pathlib import Path

import numpy as np
import pandas as pd


ORDERS_PATH = Path("datasets/orders_clean.csv")
INPUT_PATH = Path("datasets/processed/rfm_recent_features.csv")
OUTPUT_PATH = Path("datasets/processed/rfm_recent_gaps_features.csv")


def build_features(orders, base):
    results = []

    for snapshot, group in base.groupby("snapshot"):
        start = snapshot - pd.DateOffset(months=12)

        history = orders[
            (orders["OrderDate"] >= start)
            & (orders["OrderDate"] < snapshot)
        ]

        # Multiple orders on the same day count as one date.
        dates = (
            history[["StoreID", "OrderDate"]]
            .drop_duplicates()
            .sort_values(["StoreID", "OrderDate"])
            .copy()
        )

        dates["gap_days"] = (
            dates.groupby("StoreID")["OrderDate"]
            .diff()
            .dt.days
        )

        gaps = (
            dates.groupby("StoreID")
            .agg(
                mean_gap=("gap_days", "mean"),
                std_gap=("gap_days", "std"),
                distinct_dates=("OrderDate", "count"),
            )
            .reset_index()
        )

        result = group.merge(
            gaps,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        result["overdue_ratio"] = (
            result["recency_days"] / result["mean_gap"]
        )

        # Validate missing values based on distinct order dates.
        assert (
            result["mean_gap"].notna()
            == (result["distinct_dates"] >= 2)
        ).all()

        assert (
            result["std_gap"].notna()
            == (result["distinct_dates"] >= 3)
        ).all()

        results.append(result.drop(columns="distinct_dates"))

    return pd.concat(results, ignore_index=True)


def verify(features, base):
    assert len(features) == len(base)

    # Original features must remain unchanged.
    pd.testing.assert_frame_equal(
        features[base.columns]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True),
        base.sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True),
    )

    assert not features.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    for col in ["mean_gap", "std_gap", "overdue_ratio"]:
        values = features[col].dropna()

        assert np.isfinite(values).all()
        assert (values >= 0).all()

    # Manual checks from our previously verified orders.
    july = features[
        features["snapshot"] == pd.Timestamp("2012-07-01")
    ].set_index("StoreID")

    # Store 326: one historical order.
    assert pd.isna(july.loc[326, "mean_gap"])
    assert pd.isna(july.loc[326, "std_gap"])
    assert pd.isna(july.loc[326, "overdue_ratio"])

    # Store 292: gaps of 181 and 92 days.
    assert np.isclose(july.loc[292, "mean_gap"], 136.5)
    assert np.isclose(
        july.loc[292, "std_gap"],
        np.std([181, 92], ddof=1),
    )
    assert np.isclose(
        july.loc[292, "overdue_ratio"],
        62 / 136.5,
    )

    # Store 616: gaps of 92, 92 and 89 days.
    assert np.isclose(july.loc[616, "mean_gap"], 91)
    assert np.isclose(
        july.loc[616, "overdue_ratio"],
        93 / 91,
    )

    print("[PASS] Observation counts unchanged")
    print("[PASS] Original features preserved")
    print("[PASS] Missing-value rules")
    print("[PASS] No invalid gap values")
    print("[PASS] Manual reseller calculations")


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
    )

    base = pd.read_csv(
        INPUT_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 1E.5: Purchase Gap Features ===\n")

    features = build_features(orders, base)
    verify(features, base)

    columns = ["mean_gap", "std_gap", "overdue_ratio"]

    print("\nFeature summary:")
    print(features[columns].describe().round(2))

    print("\nMissing values:")
    print(features[columns].isna().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total observations: {len(features)}")
    print("\n[PASS] Stage 1E.5 completed")


if __name__ == "__main__":
    main()

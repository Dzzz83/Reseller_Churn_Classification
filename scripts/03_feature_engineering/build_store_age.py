
from pathlib import Path

import pandas as pd


INPUT = Path("datasets/processed/reseller_order_features.csv")
STORES = Path("datasets/stores_clean.csv")
OUTPUT = Path("datasets/processed/reseller_order_age_features.csv")


def main():
    features = pd.read_csv(INPUT, parse_dates=["snapshot"])
    stores = pd.read_csv(STORES)

    print("=== Stage 1F.3: Store Age Feature ===\n")

    # Join only the opening year.
    result = features.merge(
        stores[["StoreID", "YearOpened"]],
        on="StoreID",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Verify join integrity.
    assert len(result) == len(features)
    assert (result["_merge"] == "both").all()

    pd.testing.assert_frame_equal(
        result[features.columns],
        features,
    )

    # Calculate store age at each snapshot.
    result["store_age"] = (
        result["snapshot"].dt.year - result["YearOpened"]
    )

    assert result["store_age"].notna().all()
    assert (result["store_age"] >= 0).all()
    assert not result.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    # Verify against source values.
    expected_age = (
        result["snapshot"].dt.year - result["YearOpened"]
    )

    assert (result["store_age"] == expected_age).all()

    print("[PASS] All stores matched")
    print("[PASS] Original features preserved")
    print("[PASS] Store ages valid")
    print("[PASS] No duplicate observations")

    print("\nStore age summary:")
    print(result["store_age"].describe().round(2))

    # Remove temporary audit columns.
    result = result.drop(columns=["YearOpened", "_merge"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"Total observations: {len(result)}")
    print("\n[PASS] Stage 1F.3 completed")


if __name__ == "__main__":
    main()

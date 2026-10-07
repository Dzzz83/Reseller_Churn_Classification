
from pathlib import Path

import pandas as pd


STORES_PATH = Path("datasets/stores_clean.csv")
FEATURES_PATH = Path("datasets/processed/reseller_order_features.csv")

CATEGORICAL = [
    "BusinessType",
    "Specialty",
    "Brands",
    "Internet",
]

NUMERIC = [
    "AnnualSales",
    "AnnualRevenue",
    "YearOpened",
    "SquareFeet",
    "NumberEmployees",
]


def main():
    stores = pd.read_csv(STORES_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 1F.1: Store Feature Audit ===\n")

    # 1. Store integrity
    assert stores["StoreID"].is_unique
    assert stores["StoreID"].notna().all()

    required = ["StoreID"] + CATEGORICAL + NUMERIC
    assert set(required).issubset(stores.columns)

    print(f"Stores: {len(stores)}")
    print(f"Feature observations: {len(features)}")
    print("[PASS] Store schema and identifiers")

    # 2. Categorical values
    print("\nCategorical features")
    print("-" * 45)

    for column in CATEGORICAL:
        print(f"\n{column}:")
        print(stores[column].value_counts(dropna=False).to_string())

    # 3. Numeric attributes
    print("\nNumeric features")
    print("-" * 45)
    print(stores[NUMERIC].describe().round(2))

    print("\nMissing values")
    print(stores[required].isna().sum())

    assert stores[required].notna().all().all()
    assert (stores["YearOpened"] > 0).all()
    assert (stores["AnnualSales"] >= 0).all()
    assert (stores["AnnualRevenue"] >= 0).all()
    assert (stores["SquareFeet"] >= 0).all()
    assert (stores["NumberEmployees"] >= 0).all()

    print("[PASS] Missing-value and range checks")

    # 4. Verify join without changing source files
    joined = features.merge(
        stores[required],
        on="StoreID",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    assert len(joined) == len(features)
    assert (joined["_merge"] == "both").all()

    # 5. Check historical consistency
    future_openings = (
        joined["YearOpened"] > joined["snapshot"].dt.year
    )

    assert not future_openings.any()

    print("\nJoin verification")
    print("-" * 45)
    print(f"Original observations: {len(features)}")
    print(f"Joined observations:   {len(joined)}")
    print(f"Unmatched stores:      {(joined['_merge'] != 'both').sum()}")
    print(f"Future opening dates:  {future_openings.sum()}")

    print("\n[PASS] Store demographic audit completed")
    print("No files modified.")


if __name__ == "__main__":
    main()

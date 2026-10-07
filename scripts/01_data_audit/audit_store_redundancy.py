
import numpy as np
import pandas as pd


DATA_PATH = "datasets/stores_clean.csv"

NUMERIC_COLUMNS = [
    "AnnualSales",
    "AnnualRevenue",
    "SquareFeet",
    "NumberEmployees",
]


def main():
    stores = pd.read_csv(DATA_PATH)

    print("=== Stage 1F.2: Store Redundancy Audit ===\n")

    # 1. Check the suspected 10:1 relationship.
    matches = np.isclose(
        stores["AnnualSales"],
        stores["AnnualRevenue"] * 10,
        rtol=0,
        atol=0.01,
    )

    print("AnnualSales vs AnnualRevenue")
    print("-" * 45)
    print(f"Matching stores: {matches.sum()}/{len(stores)}")
    print(f"Mismatching stores: {(~matches).sum()}")

    # 2. Spearman correlation between numeric attributes.
    correlations = stores[NUMERIC_COLUMNS].corr(
        method="spearman"
    )

    print("\nSpearman correlation matrix:")
    print(correlations.round(3).to_string())

    # 3. Identify strongly correlated feature pairs.
    print("\nHighly correlated pairs (|rho| > 0.80):")

    found = False

    for i, col1 in enumerate(NUMERIC_COLUMNS):
        for col2 in NUMERIC_COLUMNS[i + 1:]:
            rho = correlations.loc[col1, col2]

            if abs(rho) > 0.80:
                print(f"{col1} <-> {col2}: {rho:.3f}")
                found = True

    if not found:
        print("None")

    # 4. Basic verification.
    assert stores["StoreID"].is_unique
    assert correlations.notna().all().all()

    print("\n[PASS] Store redundancy audit completed")
    print("No features removed.")


if __name__ == "__main__":
    main()

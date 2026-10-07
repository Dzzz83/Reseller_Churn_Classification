
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "datasets/processed/reseller_order_age_features.csv"
)

TRAIN_SNAPSHOT = pd.Timestamp("2012-07-01")
THRESHOLD = 0.80


def main():
    df = pd.read_csv(INPUT, parse_dates=["snapshot"])

    # Use only the first training snapshot.
    train = df[df["snapshot"] == TRAIN_SNAPSHOT].copy()

    assert len(train) == 326
    assert train["StoreID"].is_unique
    assert "churn" not in train.columns

    # Exclude identifiers and timestamps.
    features = train.drop(
        columns=["StoreID", "snapshot"]
    )

    numeric = features.select_dtypes(
        include="number"
    )

    print("=== Stage 1G.1: Feature Correlation ===\n")
    print(f"Training observations: {len(train)}")
    print(f"Numeric features: {len(numeric.columns)}")

    # Missing-value audit.
    print("\nMissing values (%)")
    print("-" * 45)

    missing = (
        numeric.isna().mean() * 100
    ).sort_values(ascending=False)

    print(missing.round(2).to_string())

    # Detect constant features.
    constant = [
        col for col in numeric
        if numeric[col].nunique(dropna=True) <= 1
    ]

    print("\nConstant features:")
    print(constant if constant else "None")

    # Spearman correlation.
    correlations = numeric.corr(
        method="spearman"
    )

    # Identify highly correlated pairs.
    pairs = []

    columns = correlations.columns

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            rho = correlations.loc[col1, col2]

            if pd.notna(rho) and abs(rho) > THRESHOLD:
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "rho": rho,
                    "abs_rho": abs(rho),
                })

    pairs = sorted(
        pairs,
        key=lambda x: x["abs_rho"],
        reverse=True,
    )

    print(f"\nHighly correlated pairs (|rho| > {THRESHOLD})")
    print("-" * 65)

    if pairs:
        for pair in pairs:
            print(
                f"{pair['feature_1']:<25} "
                f"{pair['feature_2']:<25} "
                f"{pair['rho']:>7.3f}"
            )
    else:
        print("None")

    print(f"\nTotal highly correlated pairs: {len(pairs)}")

    # Verification.
    assert len(numeric) == 326
    assert not numeric.empty
    assert numeric.replace(
        [np.inf, -np.inf], np.nan
    ).notna().equals(numeric.notna())

    print("\n[PASS] Correlation audit completed")
    print("No features removed.")


if __name__ == "__main__":
    main()

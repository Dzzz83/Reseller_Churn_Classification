from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

TRAIN_DATES = [
    "2012-07-01",
    "2012-10-01",
]

LOG_FEATURES = [
    "recency_days",
    "revenue_12m",
    "revenue_6m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
]


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    train = df[
        df["snapshot"].isin(pd.to_datetime(TRAIN_DATES))
    ].copy()

    print("=== Stage 2.2.2: Numeric Transformation Test ===\n")
    print(f"Training observations: {len(train)}")

    results = []

    for feature in LOG_FEATURES:
        original = train[feature]

        # log1p keeps NaN values as NaN.
        transformed = np.log1p(original)

        # Existing non-missing values must remain finite.
        valid = transformed.dropna()

        assert np.isfinite(valid).all()

        # Missing-value structure must not change.
        assert transformed.isna().equals(
            original.isna()
        )

        # log1p requires values >= 0.
        assert (original.dropna() >= 0).all()

        original_skew = original.dropna().skew()
        transformed_skew = transformed.dropna().skew()

        results.append({
            "feature": feature,
            "original_skew": original_skew,
            "log_skew": transformed_skew,
            "original_min": original.min(),
            "original_max": original.max(),
            "missing": original.isna().sum(),
        })

    summary = pd.DataFrame(results)

    print("\nSkewness before and after log1p")
    print("-" * 70)

    print(
        summary[
            [
                "feature",
                "original_skew",
                "log_skew",
                "missing",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSkewness improvement")
    print("-" * 70)

    for _, row in summary.iterrows():
        before = abs(row["original_skew"])
        after = abs(row["log_skew"])

        improved = after < before

        print(
            f"{row['feature']:<20} "
            f"{before:.3f} -> {after:.3f} "
            f"{'[PASS]' if improved else '[CHECK]'}"
        )

    # Dataset itself must remain unchanged.
    assert len(train) == 692

    print("\n[PASS] Transformations generated no invalid values")
    print("[PASS] Missing-value structure preserved")
    print("No files modified.")


if __name__ == "__main__":
    main()
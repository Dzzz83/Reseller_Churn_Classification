from pathlib import Path

import numpy as np
import pandas as pd

from define_feature_sets import FEATURE_SETS

DATA_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

TRAIN_DATES = [
    "2012-07-01",
    "2012-10-01",
]

def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    train = df[
        df["snapshot"].isin(pd.to_datetime(TRAIN_DATES))
    ].copy()

    print("=== Stage 2.2.1: Numeric Distribution Audit ===\n")
    print(f"Training observations: {len(train)}")

    columns = FEATURE_SETS["full"]

    rows = []

    for col in columns:
        s = train[col].dropna()

        rows.append({
            "feature": col,
            "count": len(s),
            "missing_pct": train[col].isna().mean() * 100,
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "mean": s.mean(),
            "skew": s.skew(),
            "zero_pct": (s == 0).mean() * 100,
        })

    summary = pd.DataFrame(rows)

    print(
        summary[
            [
                "feature",
                "missing_pct",
                "min",
                "median",
                "max",
                "skew",
                "zero_pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # Suggested log candidates:
    # non-negative features with substantial positive skew.
    log_candidates = summary[
        (summary["min"] >= 0)
        & (summary["skew"] > 1)
    ]["feature"].tolist()

    print("\nPotential log1p candidates:")
    if log_candidates:
        for col in log_candidates:
            print(f"  {col}")
    else:
        print("  None")

    # Features where log1p would be inappropriate.
    signed_features = summary[
        summary["min"] < 0
    ]["feature"].tolist()

    print("\nFeatures containing negative values:")
    if signed_features:
        for col in signed_features:
            print(f"  {col}")
    else:
        print("  None")

    assert len(train) == 692
    assert not np.isinf(
        train[columns].to_numpy(dtype=float)
    ).any()

    print("\n[PASS] Numeric distribution audit completed")
    print("No transformations applied.")

if __name__ == "__main__":
    main()
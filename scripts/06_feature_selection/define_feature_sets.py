
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "datasets/processed/reseller_order_age_features.csv"
)

RFM = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
]

FULL = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
    "n_orders_6m",
    "n_orders_3m",
    "revenue_6m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
    "overdue_ratio",
    "has_previous_6m_revenue",
    "revenue_trend",
    "share_bikes",
    "share_components",
    "share_clothing",
    "share_accessories",
    "store_age",
]

REDUNDANT = {
    "revenue_6m",
    "overdue_ratio",
    "share_components",
}

REDUCED = [col for col in FULL if col not in REDUNDANT]

FEATURE_SETS = {
    "rfm": RFM,
    "full": FULL,
    "reduced": REDUCED,
}


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["snapshot"])

    print("=== Stage 1G.3: Feature Set Verification ===\n")

    assert len(df) == 2312
    assert not df.duplicated(["StoreID", "snapshot"]).any()

    assert len(RFM) == 3
    assert len(FULL) == 17
    assert len(REDUCED) == 14

    for name, columns in FEATURE_SETS.items():
        assert len(columns) == len(set(columns))
        assert set(columns).issubset(df.columns)
        assert "churn" not in columns
        assert "StoreID" not in columns
        assert "snapshot" not in columns

        X = df[columns]

        assert all(
            pd.api.types.is_numeric_dtype(X[col])
            for col in X.columns
        )
        assert not np.isinf(X.to_numpy(dtype=float)).any()

        print(f"{name.upper()}: {len(columns)} features")
        print(", ".join(columns))
        print("[PASS] Schema and numeric integrity\n")

    assert set(RFM).issubset(REDUCED)
    assert set(REDUCED).issubset(FULL)

    # Audit missingness on training data only.
    for fold, dates in {
        "fold_1": ["2012-07-01"],
        "fold_2": ["2012-07-01", "2012-10-01"],
    }.items():
        train = df[
            df["snapshot"].isin(pd.to_datetime(dates))
        ]

        print(f"{fold}: {len(train)} training observations")

        for name, columns in FEATURE_SETS.items():
            missing = train[columns].isna().sum().sum()
            print(f"  {name}: {missing} missing cells")

    print("\n[PASS] All feature configurations verified")
    print("No source files modified.")


if __name__ == "__main__":
    main()

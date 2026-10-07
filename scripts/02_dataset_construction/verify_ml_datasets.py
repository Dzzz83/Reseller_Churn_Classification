
from pathlib import Path

import numpy as np
import pandas as pd

from compare_split_strategies import (
    STRATEGIES,
    FINAL_TRAIN_DATES,
    TEST_DATE,
)
from define_feature_sets import FEATURE_SETS


DATA_DIR = Path("datasets/processed")


def verify_dataset(df, name):
    assert not df.empty
    assert df[["StoreID", "snapshot"]].notna().all().all()
    assert not df.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    print(f"[PASS] {name}: unique reseller-snapshot pairs")


def verify_splits(train):
    print("\nValidation strategy verification")
    print("-" * 55)

    for strategy, folds in STRATEGIES.items():
        print(f"\n{strategy.upper()}")

        for i, fold in enumerate(folds, start=1):
            val_date = pd.Timestamp(fold["validation"])

            training = train[
                train["snapshot"].isin(
                    pd.to_datetime(fold["train"])
                )
            ]

            validation = train[
                train["snapshot"] == val_date
            ]

            assert not training.empty
            assert not validation.empty

            # Every training label must be known
            # before validation begins.
            for date in fold["train"]:
                label_end = (
                    pd.Timestamp(date)
                    + pd.DateOffset(months=6)
                )
                assert label_end <= val_date

            # Validation labels must be known
            # before the final test.
            assert (
                val_date + pd.DateOffset(months=6)
                <= pd.Timestamp(TEST_DATE)
            )

            # No overlapping reseller-snapshot keys.
            train_keys = set(
                zip(training["StoreID"], training["snapshot"])
            )
            val_keys = set(
                zip(validation["StoreID"], validation["snapshot"])
            )

            assert train_keys.isdisjoint(val_keys)

            print(
                f"Fold {i}: "
                f"Train={len(training)}, "
                f"Validation={len(validation)}, "
                f"Churners={validation['churn'].sum()}"
            )

            print("[PASS] Chronological separation")


def main():
    train = pd.read_csv(
        DATA_DIR / "ml_labeled_snapshots.csv",
        parse_dates=["snapshot"],
    )

    test = pd.read_csv(
        DATA_DIR / "ml_test_features.csv",
        parse_dates=["snapshot"],
    )

    print("=== Stage 1H.2: Final Dataset Verification ===\n")

    verify_dataset(train, "Historical dataset")
    verify_dataset(test, "Final test dataset")

    assert len(train) == 1375
    assert len(test) == 492

    assert int(train["churn"].sum()) == 212
    assert train["churn"].isin([0, 1]).all()

    assert "churn" not in test.columns
    assert "future_orders" not in train.columns
    assert "future_orders" not in test.columns

    print("[PASS] Row counts and labels")
    print("[PASS] Test labels excluded")

    # Verify the feature schema.
    train_features = train.drop(columns="churn")
    assert list(train_features.columns) == list(test.columns)

    for name, columns in FEATURE_SETS.items():
        for column in columns:
            assert column in train_features.columns

        assert not np.isinf(
            train_features[columns].to_numpy(dtype=float)
        ).any()

        print(f"[PASS] {name}: {len(columns)} features")

    # Check final training chronology.
    actual_dates = set(train["snapshot"])
    expected_dates = set(pd.to_datetime(FINAL_TRAIN_DATES))

    assert actual_dates == expected_dates
    assert test["snapshot"].nunique() == 1
    assert (
        test["snapshot"].iloc[0]
        == pd.Timestamp(TEST_DATE)
    )

    for date in actual_dates:
        assert (
            date + pd.DateOffset(months=6)
            <= pd.Timestamp(TEST_DATE)
        )

    print("[PASS] Final training and test chronology")

    verify_splits(train)

    print("\n=== Final Result ===")
    print("[PASS] All dataset and split checks passed")
    print("No files modified. Test labels not accessed.")


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from define_feature_sets import FEATURE_SETS


DATA_PATH = "datasets/processed/ml_labeled_snapshots.csv"

FOLDS = {
    "fold_1": {
        "train": ["2012-07-01"],
        "validation": "2013-01-01",
    },
    "fold_2": {
        "train": ["2012-07-01", "2012-10-01"],
        "validation": "2013-04-01",
    },
}


def test_imputation(df, fold_name, fold):
    train = df[
        df["snapshot"].isin(pd.to_datetime(fold["train"]))
    ]

    validation = df[
        df["snapshot"] == pd.Timestamp(fold["validation"])
    ]

    print(f"\n=== {fold_name.upper()} ===")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    for name, columns in FEATURE_SETS.items():
        X_train = train[columns]
        X_val = validation[columns]

        # Fit using training data ONLY.
        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
        )

        train_processed = imputer.fit_transform(X_train)

        # Apply learned medians to validation.
        learned_medians = imputer.statistics_.copy()
        val_processed = imputer.transform(X_val)

        # Check fitted values against training medians.
        expected_medians = X_train.median().to_numpy()

        assert np.allclose(
            learned_medians,
            expected_medians,
        )

        # Every feature must have an observed training value.
        assert np.isfinite(learned_medians).all()

        # Verify transformation.
        assert train_processed.shape[0] == len(train)
        assert val_processed.shape[0] == len(validation)

        assert np.isfinite(train_processed).all()
        assert np.isfinite(val_processed).all()

        # Validation must not modify fitted statistics.
        assert np.array_equal(
            learned_medians,
            imputer.statistics_,
        )

        indicators = len(imputer.indicator_.features_)

        print(f"\n{name.upper()}")
        print(f"  Original features: {len(columns)}")
        print(f"  Missing indicators: {indicators}")
        print(f"  Output features: {train_processed.shape[1]}")
        print(f"  Training missing cells: {X_train.isna().sum().sum()}")
        print(f"  Validation missing cells: {X_val.isna().sum().sum()}")

        print("  [PASS] No missing values after preprocessing")
        print("  [PASS] Training-only imputation")


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 2.1: Missing-Value Imputation ===")

    for fold_name, fold in FOLDS.items():
        test_imputation(df, fold_name, fold)

    print("\n[PASS] Stage 2.1 tests completed")
    print("No files modified. Final test not accessed.")


if __name__ == "__main__":
    main()

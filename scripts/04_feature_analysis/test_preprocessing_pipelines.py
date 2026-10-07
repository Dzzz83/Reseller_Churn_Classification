import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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

LOG_FEATURES = {
    "recency_days",
    "revenue_12m",
    "revenue_6m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
}


def build_tree_preprocessor(columns):
    """
    Random Forest preprocessing:
    - median imputation
    - missing indicators
    - no scaling
    - no log transformation
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ])


def build_linear_basic_preprocessor(columns):
    """
    Logistic Regression baseline:
    - median imputation
    - missing indicators
    - standard scaling
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])


def build_linear_log_preprocessor(columns):
    """
    Logistic Regression with selective log1p transformation.
    """

    log_columns = [
        col for col in columns
        if col in LOG_FEATURES
    ]

    raw_columns = [
        col for col in columns
        if col not in LOG_FEATURES
    ]

    transformers = []

    if log_columns:
        log_pipeline = Pipeline([
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])

        transformers.append(
            ("log", log_pipeline, log_columns)
        )

    if raw_columns:
        raw_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])

        transformers.append(
            ("raw", raw_pipeline, raw_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def verify_output(name, X_train_processed, X_val_processed):
    assert X_train_processed.shape[0] > 0
    assert X_val_processed.shape[0] > 0

    assert (
        X_train_processed.shape[1]
        == X_val_processed.shape[1]
    )

    assert np.isfinite(X_train_processed).all()
    assert np.isfinite(X_val_processed).all()

    print(
        f"  {name:<15} "
        f"output features: "
        f"{X_train_processed.shape[1]}"
    )
    print("  [PASS] finite values")
    print("  [PASS] train/validation schema match")


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 2.3: Preprocessing Pipeline Test ===")

    for fold_name, fold in FOLDS.items():
        train = df[
            df["snapshot"].isin(
                pd.to_datetime(fold["train"])
            )
        ]

        validation = df[
            df["snapshot"]
            == pd.Timestamp(fold["validation"])
        ]

        print(f"\n{'=' * 60}")
        print(fold_name.upper())
        print("=" * 60)

        print(f"Train:      {len(train)}")
        print(f"Validation: {len(validation)}")

        for feature_set_name, columns in FEATURE_SETS.items():
            print(f"\n{feature_set_name.upper()}")

            X_train = train[columns]
            X_val = validation[columns]

            preprocessors = {
                "tree": build_tree_preprocessor(columns),
                "linear_basic":
                    build_linear_basic_preprocessor(columns),
                "linear_log":
                    build_linear_log_preprocessor(columns),
            }

            for name, preprocessor in preprocessors.items():

                # Fit on TRAINING ONLY.
                X_train_processed = (
                    preprocessor.fit_transform(X_train)
                )

                X_val_processed = (
                    preprocessor.transform(X_val)
                )

                verify_output(
                    name,
                    X_train_processed,
                    X_val_processed,
                )

    print("\n[PASS] All preprocessing pipelines verified")
    print("Final test dataset was not accessed.")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from define_feature_sets import FEATURE_SETS
from test_preprocessing_pipelines import (
    build_tree_preprocessor,
)


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

FEATURES = FEATURE_SETS["reduced"]

FOLDS = {
    "fold_1": {
        "train": ["2012-07-01"],
        "validation": "2013-01-01",
    },
    "fold_2": {
        "train": [
            "2012-07-01",
            "2012-10-01",
        ],
        "validation": "2013-04-01",
    },
}


def run_fold(train, validation):
    X_train = train[FEATURES]
    y_train = train["churn"]

    X_val = validation[FEATURES]
    y_val = validation["churn"]

    pipeline = Pipeline([
        (
            "preprocessor",
            build_tree_preprocessor(FEATURES),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    pipeline.fit(
        X_train,
        y_train,
    )

    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=50,
        random_state=42,
        n_jobs=-1,
    )

    return pd.DataFrame({
        "feature": FEATURES,
        "importance_mean":
            result.importances_mean,
        "importance_std":
            result.importances_std,
    })


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print(
        "=== Permutation Importance Verification ===\n"
    )

    fold_results = []

    for fold_name, fold in FOLDS.items():

        train = df[
            df["snapshot"].isin(
                pd.to_datetime(
                    fold["train"]
                )
            )
        ]

        validation = df[
            df["snapshot"]
            == pd.Timestamp(
                fold["validation"]
            )
        ]

        result = run_fold(
            train,
            validation,
        )

        result["fold"] = fold_name

        fold_results.append(result)

        print(f"{fold_name.upper()}")
        print(
            result.sort_values(
                "importance_mean",
                ascending=False,
            )
            .round(4)
            .to_string(index=False)
        )

        print()

    combined = pd.concat(
        fold_results,
        ignore_index=True,
    )

    summary = (
        combined
        .pivot(
            index="feature",
            columns="fold",
            values="importance_mean",
        )
        .reset_index()
    )

    summary["mean_importance"] = (
        summary[
            ["fold_1", "fold_2"]
        ].mean(axis=1)
    )

    summary["positive_both_folds"] = (
        (summary["fold_1"] > 0)
        & (summary["fold_2"] > 0)
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    )

    print(
        "=== Cross-Fold Summary ==="
    )

    print(
        summary.round(4)
        .to_string(index=False)
    )

    print(
        "\nConsistently positive features"
    )
    print("-" * 50)

    stable = summary[
        summary["positive_both_folds"]
    ]

    if stable.empty:
        print("None")
    else:
        for _, row in stable.iterrows():
            print(
                f"{row['feature']:<28} "
                f"Fold1={row['fold_1']:+.4f} "
                f"Fold2={row['fold_2']:+.4f} "
                f"Mean={row['mean_importance']:+.4f}"
            )

    print(
        "\n[PASS] Permutation importance completed"
    )
    print(
        "Validation folds only. "
        "Final test data not accessed."
    )


if __name__ == "__main__":
    main()
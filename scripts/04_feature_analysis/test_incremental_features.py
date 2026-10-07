from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from test_preprocessing_pipelines import (
    build_tree_preprocessor,
)
from define_feature_sets import FULL


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

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

CORE = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
]

CANDIDATES = [
    feature
    for feature in FULL
    if feature not in CORE
]


def evaluate_feature_set(
    train,
    validation,
    features,
):
    X_train = train[features]
    y_train = train["churn"]

    X_val = validation[features]
    y_val = validation["churn"]

    preprocessor = build_tree_preprocessor(
        features
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_val_processed = (
        preprocessor.transform(X_val)
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    return average_precision_score(
        y_val,
        probabilities,
    )


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print(
        "=== Incremental Feature Usefulness: Add-One Test ===\n"
    )

    # --------------------------------------------------
    # Core baseline
    # --------------------------------------------------
    core_scores = {}

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

        core_scores[fold_name] = (
            evaluate_feature_set(
                train,
                validation,
                CORE,
            )
        )

    core_mean = sum(
        core_scores.values()
    ) / len(core_scores)

    print("CORE FEATURES:")
    print(", ".join(CORE))

    print(
        f"\nCore Fold 1 PR-AUC: "
        f"{core_scores['fold_1']:.3f}"
    )

    print(
        f"Core Fold 2 PR-AUC: "
        f"{core_scores['fold_2']:.3f}"
    )

    print(
        f"Core Mean PR-AUC:   "
        f"{core_mean:.3f}"
    )

    # --------------------------------------------------
    # Add one candidate feature
    # --------------------------------------------------
    results = []

    for candidate in CANDIDATES:
        features = CORE + [candidate]

        fold_scores = {}

        for fold_name, fold in FOLDS.items():
            train = df[
                df["snapshot"].isin(
                    pd.to_datetime(fold["train"])
                )
            ]

            validation = df[
                df["snapshot"]
                == pd.Timestamp(
                    fold["validation"]
                )
            ]

            fold_scores[fold_name] = (
                evaluate_feature_set(
                    train,
                    validation,
                    features,
                )
            )

        mean_score = sum(
            fold_scores.values()
        ) / len(fold_scores)

        results.append({
            "feature": candidate,
            "fold_1": fold_scores["fold_1"],
            "fold_2": fold_scores["fold_2"],
            "mean_pr_auc": mean_score,
            "delta_vs_core":
                mean_score - core_mean,
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "delta_vs_core",
        ascending=False,
    )

    print(
        "\n=== Add-One Results ==="
    )

    print(
        results.round(3).to_string(
            index=False
        )
    )

    print(
        "\nFeatures that improved "
        "mean PR-AUC:"
    )

    improved = results[
        results["delta_vs_core"] > 0
    ]

    if improved.empty:
        print("None")
    else:
        for _, row in improved.iterrows():
            print(
                f"  {row['feature']:<28} "
                f"{row['delta_vs_core']:+.3f}"
            )

    print(
        "\n[PASS] Incremental feature test completed"
    )

    print(
        "Validation folds only. "
        "Final test data not accessed."
    )


if __name__ == "__main__":
    main()
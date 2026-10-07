from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from define_feature_sets import FEATURE_SETS
from test_preprocessing_pipelines import (
    build_tree_preprocessor,
)


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

REDUCED_FEATURES = FEATURE_SETS["reduced"]

PRUNED_FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]

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

SEEDS = list(range(20))


def evaluate(
    train,
    validation,
    features,
    seed,
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
        random_state=seed,
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
        "=== Reduced vs Pruned Feature Set ===\n"
    )

    print(
        f"Reduced features: {len(REDUCED_FEATURES)}"
    )
    print(
        f"Pruned features:  {len(PRUNED_FEATURES)}"
    )

    results = []

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

        for seed in SEEDS:

            reduced_score = evaluate(
                train,
                validation,
                REDUCED_FEATURES,
                seed,
            )

            pruned_score = evaluate(
                train,
                validation,
                PRUNED_FEATURES,
                seed,
            )

            results.append({
                "fold": fold_name,
                "seed": seed,
                "reduced_pr_auc":
                    reduced_score,
                "pruned_pr_auc":
                    pruned_score,
                "delta":
                    pruned_score
                    - reduced_score,
            })

    results = pd.DataFrame(results)

    print("\n=== Per-Fold Summary ===")

    fold_summary = (
        results
        .groupby("fold")
        .agg(
            reduced_mean=(
                "reduced_pr_auc",
                "mean",
            ),
            reduced_std=(
                "reduced_pr_auc",
                "std",
            ),
            pruned_mean=(
                "pruned_pr_auc",
                "mean",
            ),
            pruned_std=(
                "pruned_pr_auc",
                "std",
            ),
            mean_delta=(
                "delta",
                "mean",
            ),
            delta_std=(
                "delta",
                "std",
            ),
        )
        .reset_index()
    )

    print(
        fold_summary
        .round(4)
        .to_string(index=False)
    )

    print("\n=== Win Rate ===")

    for fold_name in FOLDS:
        fold_results = results[
            results["fold"] == fold_name
        ]

        wins = (
            fold_results["delta"] > 0
        ).sum()

        ties = (
            fold_results["delta"] == 0
        ).sum()

        losses = (
            fold_results["delta"] < 0
        ).sum()

        print(
            f"{fold_name}: "
            f"Pruned wins {wins}/20, "
            f"ties {ties}, "
            f"losses {losses}"
        )

    overall_reduced = (
        results["reduced_pr_auc"].mean()
    )

    overall_pruned = (
        results["pruned_pr_auc"].mean()
    )

    overall_delta = (
        results["delta"].mean()
    )

    print("\n=== Overall ===")

    print(
        f"Reduced mean PR-AUC: "
        f"{overall_reduced:.4f}"
    )

    print(
        f"Pruned mean PR-AUC:  "
        f"{overall_pruned:.4f}"
    )

    print(
        f"Mean improvement:     "
        f"{overall_delta:+.4f}"
    )

    print(
        "\n[PASS] Feature-set comparison completed"
    )

    print(
        "Validation folds only. "
        "Final test data not accessed."
    )


if __name__ == "__main__":
    main()
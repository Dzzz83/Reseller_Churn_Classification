from pathlib import Path

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

BASE_FEATURES = FEATURE_SETS["reduced"]


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


def evaluate_across_folds(df, features):
    scores = {}

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

        scores[fold_name] = evaluate_feature_set(
            train,
            validation,
            features,
        )

    mean_score = sum(scores.values()) / len(scores)

    return scores, mean_score


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Drop-One Feature Usefulness Test ===\n")

    print(f"Base features: {len(BASE_FEATURES)}")
    print(", ".join(BASE_FEATURES))

    # --------------------------------------------------
    # Baseline with all reduced features
    # --------------------------------------------------
    base_scores, base_mean = evaluate_across_folds(
        df,
        BASE_FEATURES,
    )

    print("\nBASELINE")
    print(
        f"Fold 1 PR-AUC: {base_scores['fold_1']:.3f}"
    )
    print(
        f"Fold 2 PR-AUC: {base_scores['fold_2']:.3f}"
    )
    print(
        f"Mean PR-AUC:   {base_mean:.3f}"
    )

    # --------------------------------------------------
    # Remove one feature at a time
    # --------------------------------------------------
    results = []

    for removed_feature in BASE_FEATURES:
        remaining = [
            feature
            for feature in BASE_FEATURES
            if feature != removed_feature
        ]

        scores, mean_score = evaluate_across_folds(
            df,
            remaining,
        )

        results.append({
            "removed_feature": removed_feature,
            "fold_1": scores["fold_1"],
            "fold_2": scores["fold_2"],
            "mean_pr_auc": mean_score,

            # Positive value means removal improved model.
            "delta_vs_base":
                mean_score - base_mean,
        })

    results = pd.DataFrame(results)

    # Most harmful removals first.
    results = results.sort_values(
        "delta_vs_base",
        ascending=True,
    )

    print("\n=== Drop-One Results ===")

    print(
        results.round(3).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------
    print("\nFeatures whose removal HURT performance")
    print("-" * 55)

    useful = results[
        results["delta_vs_base"] < 0
    ]

    if useful.empty:
        print("None")
    else:
        for _, row in useful.iterrows():
            print(
                f"{row['removed_feature']:<28} "
                f"{row['delta_vs_base']:+.3f}"
            )

    print("\nFeatures whose removal IMPROVED performance")
    print("-" * 55)

    harmful = results[
        results["delta_vs_base"] > 0
    ]

    if harmful.empty:
        print("None")
    else:
        for _, row in harmful.iterrows():
            print(
                f"{row['removed_feature']:<28} "
                f"{row['delta_vs_base']:+.3f}"
            )

    print(
        "\n[PASS] Drop-one feature test completed"
    )

    print(
        "Rolling validation only. "
        "Final test data not accessed."
    )


if __name__ == "__main__":
    main()
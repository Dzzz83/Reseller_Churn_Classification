from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

FEATURES = [
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
        "train": ["2012-07-01", "2012-10-01"],
        "validation": "2013-04-01",
    },
}

SEEDS = range(20)
THRESHOLD = 0.5


def build_rf(seed):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gb(seed):
    return HistGradientBoostingClassifier(
        random_state=seed,
    )


def random_oversample(x, y, seed):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    combined = x.copy()
    combined["_target"] = y

    majority = combined[combined["_target"] == 0]
    minority = combined[combined["_target"] == 1]

    minority_oversampled = minority.sample(
        n=len(majority),
        replace=True,
        random_state=seed,
    )

    balanced = pd.concat(
        [majority, minority_oversampled],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    y_resampled = balanced.pop("_target").astype(int)

    return balanced, y_resampled


def evaluate(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    df["snapshot"] = pd.to_datetime(
        df["snapshot"]
    ).dt.strftime("%Y-%m-%d")

    results = []

    for fold_name, fold in FOLDS.items():

        train = df[
            df["snapshot"].isin(fold["train"])
        ].copy()

        val = df[
            df["snapshot"] == fold["validation"]
        ].copy()

        x_train = train[FEATURES]
        y_train = train["churn"].astype(int)

        x_val = val[FEATURES]
        y_val = val["churn"].astype(int)

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        print(f"\n=== {fold_name} ===")
        print(
            f"Original training: {len(y_train)} "
            f"(churn={y_train.sum()}, "
            f"non-churn={(y_train == 0).sum()})"
        )

        # -------------------------------------------------
        # Random Forest
        # -------------------------------------------------

        for seed in SEEDS:

            # No balancing
            model = build_rf(seed)

            model.fit(
                x_train,
                y_train,
            )

            probabilities = model.predict_proba(
                x_val
            )[:, 1]

            results.append(
                {
                    "fold": fold_name,
                    "model": "Random Forest",
                    "strategy": "none",
                    "seed": seed,
                    **evaluate(
                        y_val,
                        probabilities,
                    ),
                }
            )

            # Balanced class weighting
            model = build_rf(seed)

            model.fit(
                x_train,
                y_train,
                model__sample_weight=weights,
            )

            probabilities = model.predict_proba(
                x_val
            )[:, 1]

            results.append(
                {
                    "fold": fold_name,
                    "model": "Random Forest",
                    "strategy": "balanced",
                    "seed": seed,
                    **evaluate(
                        y_val,
                        probabilities,
                    ),
                }
            )

            # Random oversampling
            x_ros, y_ros = random_oversample(
                x_train,
                y_train,
                seed,
            )

            model = build_rf(seed)

            model.fit(
                x_ros,
                y_ros,
            )

            probabilities = model.predict_proba(
                x_val
            )[:, 1]

            results.append(
                {
                    "fold": fold_name,
                    "model": "Random Forest",
                    "strategy": "oversampling",
                    "seed": seed,
                    **evaluate(
                        y_val,
                        probabilities,
                    ),
                }
            )

        # -------------------------------------------------
        # Gradient Boosting
        # -------------------------------------------------

        # No balancing
        model = build_gb(42)

        model.fit(
            x_train,
            y_train,
        )

        probabilities = model.predict_proba(
            x_val
        )[:, 1]

        results.append(
            {
                "fold": fold_name,
                "model": "Gradient Boosting",
                "strategy": "none",
                "seed": np.nan,
                **evaluate(
                    y_val,
                    probabilities,
                ),
            }
        )

        # Balanced weighting
        model = build_gb(42)

        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
        )

        probabilities = model.predict_proba(
            x_val
        )[:, 1]

        results.append(
            {
                "fold": fold_name,
                "model": "Gradient Boosting",
                "strategy": "balanced",
                "seed": np.nan,
                **evaluate(
                    y_val,
                    probabilities,
                ),
            }
        )

        # Random oversampling
        x_ros, y_ros = random_oversample(
            x_train,
            y_train,
            42,
        )

        print(
            f"Oversampled training: {len(y_ros)} "
            f"(churn={y_ros.sum()}, "
            f"non-churn={(y_ros == 0).sum()})"
        )

        model = build_gb(42)

        model.fit(
            x_ros,
            y_ros,
        )

        probabilities = model.predict_proba(
            x_val
        )[:, 1]

        results.append(
            {
                "fold": fold_name,
                "model": "Gradient Boosting",
                "strategy": "oversampling",
                "seed": np.nan,
                **evaluate(
                    y_val,
                    probabilities,
                ),
            }
        )

    results = pd.DataFrame(results)

    per_fold = (
        results
        .groupby(
            ["model", "strategy", "fold"]
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )

    per_fold["std_pr_auc"] = (
        per_fold["std_pr_auc"].fillna(0)
    )

    print("\n=== Per-Fold Results ===")

    print(
        per_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    overall = (
        per_fold
        .groupby(["model", "strategy"])
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold_pr_auc=("mean_pr_auc", "min"),
            mean_precision=("mean_precision", "mean"),
            mean_recall=("mean_recall", "mean"),
            mean_f1=("mean_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            ["model", "mean_pr_auc"],
            ascending=[True, False],
        )
    )

    print("\n=== Overall Summary ===")

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Primary decision metric: PR-AUC")
    print(
        "Precision/Recall/F1 at threshold 0.5 "
        "are diagnostic only."
    )
    print(
        "Oversampling was applied ONLY "
        "to each training fold."
    )
    print("Final test labels were NOT used.")


if __name__ == "__main__":
    main()
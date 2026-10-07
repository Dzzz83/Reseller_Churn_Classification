from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
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


def evaluate(y_true, probabilities):
    pred = (probabilities >= THRESHOLD).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    snapshot_col = "snapshot"
    target_col = "churn"

    df[snapshot_col] = pd.to_datetime(
        df[snapshot_col]
    ).dt.strftime("%Y-%m-%d")

    results = []

    for fold_name, fold in FOLDS.items():

        train = df[
            df[snapshot_col].isin(fold["train"])
        ]

        val = df[
            df[snapshot_col] == fold["validation"]
        ]

        x_train = train[FEATURES]
        y_train = train[target_col].astype(int)

        x_val = val[FEATURES]
        y_val = val[target_col].astype(int)

        balanced_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        print(f"\n=== {fold_name} ===")
        print(
            f"Train: {len(train)}, "
            f"churn={y_train.mean():.3f}"
        )
        print(
            f"Validation: {len(val)}, "
            f"churn={y_val.mean():.3f}"
        )

        # -------------------------
        # Random Forest
        # -------------------------
        for strategy in ["none", "balanced"]:

            for seed in SEEDS:

                model = build_rf(seed)

                if strategy == "balanced":
                    model.fit(
                        x_train,
                        y_train,
                        model__sample_weight=balanced_weights,
                    )
                else:
                    model.fit(
                        x_train,
                        y_train,
                    )

                probabilities = model.predict_proba(
                    x_val
                )[:, 1]

                metrics = evaluate(
                    y_val,
                    probabilities,
                )

                results.append(
                    {
                        "fold": fold_name,
                        "model": "Random Forest",
                        "strategy": strategy,
                        "seed": seed,
                        **metrics,
                    }
                )

        # -------------------------
        # Gradient Boosting
        # -------------------------
        for strategy in ["none", "balanced"]:

            model = build_gb(42)

            if strategy == "balanced":
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=balanced_weights,
                )
            else:
                model.fit(
                    x_train,
                    y_train,
                )

            probabilities = model.predict_proba(
                x_val
            )[:, 1]

            metrics = evaluate(
                y_val,
                probabilities,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": "Gradient Boosting",
                    "strategy": strategy,
                    "seed": np.nan,
                    **metrics,
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

    print("\nPrimary decision metric: PR-AUC")
    print(
        "Precision/Recall/F1 use threshold 0.5 "
        "and are diagnostic only."
    )
    print("Final test labels were NOT used.")


if __name__ == "__main__":
    main()
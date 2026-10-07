from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

FEATURES = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
    "n_orders_6m",
    "n_orders_3m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
    "has_previous_6m_revenue",
    "revenue_trend",
    "share_bikes",
    "share_clothing",
    "share_accessories",
    "store_age",
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

# Repeating seeds lets us see whether tree-model performance
# is stable rather than depending on one lucky random seed.
SEEDS = list(range(20))


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def build_logistic_regression():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_random_forest(seed):
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
                    class_weight=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting(seed):
    # HistGradientBoosting can handle NaN values directly.
    #
    # This is useful here because missing values in features such as
    # mean_gap/std_gap/revenue_trend have real meaning.
    return HistGradientBoostingClassifier(
        random_state=seed,
    )


def evaluate_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_val)[:, 1]

    return {
        "pr_auc": average_precision_score(
            y_val,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_val,
            probabilities,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    snapshot_col = find_column(
        df,
        [
            "snapshot_date",
            "snapshot",
            "SnapshotDate",
        ],
    )

    target_col = find_column(
        df,
        [
            "churn",
            "churn_label",
            "is_churn",
            "target",
        ],
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    df[snapshot_col] = pd.to_datetime(
        df[snapshot_col]
    ).dt.strftime("%Y-%m-%d")

    print("=== Model Architecture Comparison ===")
    print()
    print(f"Dataset: {DATA_PATH}")
    print(f"Features: {len(FEATURES)}")
    print(f"Target: {target_col}")
    print(f"Snapshot column: {snapshot_col}")
    print()

    results = []

    for fold_name, fold in FOLDS.items():
        train_df = df[
            df[snapshot_col].isin(fold["train"])
        ].copy()

        val_df = df[
            df[snapshot_col] == fold["validation"]
        ].copy()

        if train_df.empty or val_df.empty:
            raise ValueError(
                f"{fold_name}: train or validation data is empty."
            )

        x_train = train_df[FEATURES]
        y_train = train_df[target_col].astype(int)

        x_val = val_df[FEATURES]
        y_val = val_df[target_col].astype(int)

        prevalence = y_val.mean()

        print(f"--- {fold_name} ---")
        print(
            f"Train snapshots: {fold['train']}"
        )
        print(
            f"Validation snapshot: "
            f"{fold['validation']}"
        )
        print(
            f"Train rows: {len(train_df)} "
            f"(churn={y_train.mean():.3f})"
        )
        print(
            f"Validation rows: {len(val_df)} "
            f"(churn={prevalence:.3f})"
        )
        print(
            f"No-skill PR-AUC: {prevalence:.4f}"
        )
        print()

        # Logistic Regression is deterministic here.
        logistic = build_logistic_regression()

        metrics = evaluate_model(
            logistic,
            x_train,
            y_train,
            x_val,
            y_val,
        )

        results.append(
            {
                "fold": fold_name,
                "model": "Logistic Regression",
                "seed": np.nan,
                **metrics,
            }
        )

        # Random Forest and Gradient Boosting:
        # evaluate over multiple seeds.
        for seed in SEEDS:
            random_forest = build_random_forest(seed)

            metrics = evaluate_model(
                random_forest,
                x_train,
                y_train,
                x_val,
                y_val,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": "Random Forest",
                    "seed": seed,
                    **metrics,
                }
            )

            gradient_boosting = build_gradient_boosting(
                seed
            )

            metrics = evaluate_model(
                gradient_boosting,
                x_train,
                y_train,
                x_val,
                y_val,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": "Gradient Boosting",
                    "seed": seed,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(results)

    print()
    print("=== Per-Fold Results ===")

    per_fold = (
        results_df
        .groupby(["fold", "model"])
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            min_pr_auc=("pr_auc", "min"),
            max_pr_auc=("pr_auc", "max"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )

    # Logistic Regression has one deterministic run,
    # so std would otherwise show NaN.
    per_fold["std_pr_auc"] = (
        per_fold["std_pr_auc"].fillna(0.0)
    )

    print(
        per_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=== Overall Architecture Summary ===")

    fold_means = (
        per_fold
        .groupby("model")
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold_pr_auc=("mean_pr_auc", "min"),
            best_fold_pr_auc=("mean_pr_auc", "max"),
            mean_roc_auc=("mean_roc_auc", "mean"),
        )
        .reset_index()
        .sort_values(
            "mean_pr_auc",
            ascending=False,
        )
    )

    print(
        fold_means.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Primary decision metric: PR-AUC")
    print("Final test labels were NOT used.")


if __name__ == "__main__":
    main()
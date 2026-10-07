from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

FEATURE_SETS = {
    "RFM": [
        "recency_days",
        "n_orders_12m",
        "revenue_12m",
    ],

    "Full": [
        "recency_days",
        "n_orders_12m",
        "revenue_12m",
        "n_orders_6m",
        "n_orders_3m",
        "revenue_6m",
        "revenue_3m",
        "mean_gap",
        "std_gap",
        "overdue_ratio",
        "has_previous_6m_revenue",
        "revenue_trend",
        "share_bikes",
        "share_components",
        "share_clothing",
        "share_accessories",
        "store_age",
    ],

    "Reduced": [
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
    ],

    "Pruned": [
        "n_orders_3m",
        "revenue_3m",
        "recency_days",
        "share_bikes",
        "share_accessories",
        "share_clothing",
        "revenue_12m",
    ],
}

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

SEEDS = list(range(20))


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
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


def build_gradient_boosting():
    # HistGradientBoosting handles NaN values natively.
    return HistGradientBoostingClassifier()


def evaluate(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]

    return average_precision_score(
        y_val,
        probabilities,
    )


def main():
    df = pd.read_csv(DATA_PATH)

    snapshot_col = find_column(
        df,
        ["snapshot", "snapshot_date", "SnapshotDate"],
    )

    target_col = find_column(
        df,
        ["churn", "churn_label", "is_churn", "target"],
    )

    df[snapshot_col] = pd.to_datetime(
        df[snapshot_col]
    ).dt.strftime("%Y-%m-%d")

    all_features = sorted(
        {
            feature
            for features in FEATURE_SETS.values()
            for feature in features
        }
    )

    missing = [
        feature
        for feature in all_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    print("=== Feature Set Comparison ===")
    print("Models: Random Forest + Gradient Boosting")
    print("Primary metric: PR-AUC")
    print()

    results = []

    for fold_name, fold in FOLDS.items():
        train_df = df[
            df[snapshot_col].isin(fold["train"])
        ].copy()

        val_df = df[
            df[snapshot_col] == fold["validation"]
        ].copy()

        y_train = train_df[target_col].astype(int)
        y_val = val_df[target_col].astype(int)

        print(f"--- {fold_name} ---")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(val_df)}")
        print(f"No-skill PR-AUC: {y_val.mean():.4f}")
        print()

        for set_name, features in FEATURE_SETS.items():

            x_train = train_df[features]
            x_val = val_df[features]

            # Gradient Boosting
            gb = build_gradient_boosting()

            gb_score = evaluate(
                gb,
                x_train,
                y_train,
                x_val,
                y_val,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": "Gradient Boosting",
                    "feature_set": set_name,
                    "n_features": len(features),
                    "seed": np.nan,
                    "pr_auc": gb_score,
                }
            )

            # Random Forest: repeat across seeds
            for seed in SEEDS:
                rf = build_random_forest(seed)

                rf_score = evaluate(
                    rf,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )

                results.append(
                    {
                        "fold": fold_name,
                        "model": "Random Forest",
                        "feature_set": set_name,
                        "n_features": len(features),
                        "seed": seed,
                        "pr_auc": rf_score,
                    }
                )

    results_df = pd.DataFrame(results)

    per_fold = (
        results_df
        .groupby(
            [
                "model",
                "feature_set",
                "n_features",
                "fold",
            ]
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            min_pr_auc=("pr_auc", "min"),
            max_pr_auc=("pr_auc", "max"),
        )
        .reset_index()
    )

    per_fold["std_pr_auc"] = (
        per_fold["std_pr_auc"].fillna(0.0)
    )

    print("=== Per-Fold Results ===")
    print(
        per_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    overall = (
        per_fold
        .groupby(
            ["model", "feature_set", "n_features"]
        )
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold=("mean_pr_auc", "min"),
            best_fold=("mean_pr_auc", "max"),
        )
        .reset_index()
        .sort_values(
            ["model", "mean_pr_auc"],
            ascending=[True, False],
        )
    )

    print()
    print("=== Overall Feature-Set Summary ===")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Final test labels were NOT used.")


if __name__ == "__main__":
    main()
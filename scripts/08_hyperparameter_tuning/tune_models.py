from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path(
    "results/hyperparameter_tuning"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        "train": [
            "2012-07-01",
            "2012-10-01",
        ],
        "validation": "2013-04-01",
    },
}


SEEDS = [
    42,
    78,
    88,
    1034,
    2026,
]


RF_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", 0.7],
}


GB_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 200],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [10, 20, 30],
    "l2_regularization": [0.0, 1.0],
}


def random_oversample(x, y, seed):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    data = x.copy()
    data["_target"] = y

    non_churn = data[
        data["_target"] == 0
    ]

    churn = data[
        data["_target"] == 1
    ]

    churn_sampled = churn.sample(
        n=len(non_churn),
        replace=True,
        random_state=seed,
    )

    balanced = pd.concat(
        [non_churn, churn_sampled],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    y_balanced = balanced.pop(
        "_target"
    ).astype(int)

    return balanced, y_balanced


def build_rf(params, seed):
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
                    **params,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_gb(params, seed):
    return HistGradientBoostingClassifier(
        **params,
        random_state=seed,
        early_stopping=False,
    )


def prepare_folds(df):
    prepared = {}

    for fold_name, fold in FOLDS.items():

        train = df[
            df["snapshot"].isin(
                fold["train"]
            )
        ].copy()

        val = df[
            df["snapshot"]
            == fold["validation"]
        ].copy()

        prepared[fold_name] = {
            "x_train": train[FEATURES],
            "y_train": train[
                "churn"
            ].astype(int),
            "x_val": val[FEATURES],
            "y_val": val[
                "churn"
            ].astype(int),
        }

    return prepared


def tune_random_forest(folds):
    results = []

    grid = list(
        ParameterGrid(RF_GRID)
    )

    print()
    print(
        "=== Random Forest Tuning ==="
    )
    print(
        f"Configurations: {len(grid)}"
    )

    for index, params in enumerate(
        grid,
        start=1,
    ):

        fold_scores = {}

        for fold_name, data in folds.items():

            seed_scores = []

            for seed in SEEDS:

                model = build_rf(
                    params,
                    seed,
                )

                model.fit(
                    data["x_train"],
                    data["y_train"],
                )

                probability = (
                    model.predict_proba(
                        data["x_val"]
                    )[:, 1]
                )

                score = (
                    average_precision_score(
                        data["y_val"],
                        probability,
                    )
                )

                seed_scores.append(score)

            fold_scores[
                fold_name
            ] = np.mean(seed_scores)

        mean_score = np.mean(
            list(fold_scores.values())
        )

        worst_score = np.min(
            list(fold_scores.values())
        )

        results.append(
            {
                "model": "Random Forest",
                "params": json.dumps(
                    params
                ),
                "fold_1_pr_auc":
                    fold_scores["fold_1"],
                "fold_2_pr_auc":
                    fold_scores["fold_2"],
                "mean_pr_auc": mean_score,
                "worst_fold_pr_auc":
                    worst_score,
            }
        )

        print(
            f"[{index:02d}/{len(grid)}] "
            f"mean={mean_score:.4f} "
            f"worst={worst_score:.4f}"
        )

    return pd.DataFrame(results)


def tune_gradient_boosting(folds):
    results = []

    grid = list(
        ParameterGrid(GB_GRID)
    )

    print()
    print(
        "=== Gradient Boosting Tuning ==="
    )
    print(
        f"Configurations: {len(grid)}"
    )

    for index, params in enumerate(
        grid,
        start=1,
    ):

        fold_scores = {}

        for fold_name, data in folds.items():

            seed_scores = []

            for seed in SEEDS:

                x_train, y_train = (
                    random_oversample(
                        data["x_train"],
                        data["y_train"],
                        seed,
                    )
                )

                model = build_gb(
                    params,
                    seed,
                )

                model.fit(
                    x_train,
                    y_train,
                )

                probability = (
                    model.predict_proba(
                        data["x_val"]
                    )[:, 1]
                )

                score = (
                    average_precision_score(
                        data["y_val"],
                        probability,
                    )
                )

                seed_scores.append(score)

            fold_scores[
                fold_name
            ] = np.mean(seed_scores)

        mean_score = np.mean(
            list(fold_scores.values())
        )

        worst_score = np.min(
            list(fold_scores.values())
        )

        results.append(
            {
                "model":
                    "Gradient Boosting",
                "params": json.dumps(
                    params
                ),
                "fold_1_pr_auc":
                    fold_scores["fold_1"],
                "fold_2_pr_auc":
                    fold_scores["fold_2"],
                "mean_pr_auc": mean_score,
                "worst_fold_pr_auc":
                    worst_score,
            }
        )

        print(
            f"[{index:03d}/{len(grid)}] "
            f"mean={mean_score:.4f} "
            f"worst={worst_score:.4f}"
        )

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(DATA_PATH)

    df["snapshot"] = pd.to_datetime(
        df["snapshot"]
    ).dt.strftime("%Y-%m-%d")

    folds = prepare_folds(df)

    print(
        "=== Hyperparameter Tuning ==="
    )
    print(
        "Primary metric: PR-AUC"
    )
    print(
        "Final test labels: NOT USED"
    )

    rf_results = tune_random_forest(
        folds
    )

    gb_results = tune_gradient_boosting(
        folds
    )

    rf_results = rf_results.sort_values(
        [
            "mean_pr_auc",
            "worst_fold_pr_auc",
        ],
        ascending=False,
    )

    gb_results = gb_results.sort_values(
        [
            "mean_pr_auc",
            "worst_fold_pr_auc",
        ],
        ascending=False,
    )

    rf_results.to_csv(
        RESULTS_DIR
        / "random_forest_tuning.csv",
        index=False,
    )

    gb_results.to_csv(
        RESULTS_DIR
        / "gradient_boosting_tuning.csv",
        index=False,
    )

    print()
    print(
        "=== Top Random Forest Configurations ==="
    )

    print(
        rf_results.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== Top Gradient Boosting Configurations ==="
    )

    print(
        gb_results.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "Final test labels were NOT used."
    )


if __name__ == "__main__":
    main()
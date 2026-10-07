from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path("results/threshold_selection")
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


SEEDS = [42, 78, 88, 1034, 2026]


RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


GB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}


THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01,
)


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
                    **RF_PARAMS,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_gb(seed):
    return HistGradientBoostingClassifier(
        **GB_PARAMS,
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
            "y_train": train["churn"].astype(int),
            "x_val": val[FEATURES],
            "y_val": val["churn"].astype(int),
        }

    return prepared


def get_rf_probabilities(data):
    probabilities = []

    for seed in SEEDS:
        model = build_rf(seed)

        model.fit(
            data["x_train"],
            data["y_train"],
        )

        probabilities.append(
            model.predict_proba(
                data["x_val"]
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def get_gb_probabilities(data):
    probabilities = []

    for seed in SEEDS:

        x_train, y_train = random_oversample(
            data["x_train"],
            data["y_train"],
            seed,
        )

        model = build_gb(seed)

        model.fit(
            x_train,
            y_train,
        )

        probabilities.append(
            model.predict_proba(
                data["x_val"]
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def evaluate_model(
    model_name,
    folds,
    probability_function,
):
    fold_predictions = {}

    for fold_name, data in folds.items():

        probabilities = probability_function(data)

        fold_predictions[fold_name] = {
            "y_true": data["y_val"].to_numpy(),
            "probability": probabilities,
        }

    rows = []

    for threshold in THRESHOLDS:

        result = {
            "model": model_name,
            "threshold": threshold,
        }

        fold_metrics = []

        for fold_name, prediction in (
            fold_predictions.items()
        ):

            y_true = prediction["y_true"]

            y_pred = (
                prediction["probability"]
                >= threshold
            ).astype(int)

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            accuracy = accuracy_score(
                y_true,
                y_pred,
            )

            result[
                f"{fold_name}_precision"
            ] = precision

            result[
                f"{fold_name}_recall"
            ] = recall

            result[
                f"{fold_name}_f1"
            ] = f1

            result[
                f"{fold_name}_accuracy"
            ] = accuracy

            fold_metrics.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        result["mean_precision"] = np.mean(
            [
                x["precision"]
                for x in fold_metrics
            ]
        )

        result["mean_recall"] = np.mean(
            [
                x["recall"]
                for x in fold_metrics
            ]
        )

        result["mean_f1"] = np.mean(
            [
                x["f1"]
                for x in fold_metrics
            ]
        )

        result["worst_fold_f1"] = np.min(
            [
                x["f1"]
                for x in fold_metrics
            ]
        )

        rows.append(result)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)

    df["snapshot"] = pd.to_datetime(
        df["snapshot"]
    ).dt.strftime("%Y-%m-%d")

    folds = prepare_folds(df)

    print(
        "=== Stage 09: Threshold Selection ==="
    )
    print(
        "Hyperparameters: LOCKED"
    )
    print(
        "Final test labels: NOT USED"
    )

    print()
    print("Evaluating Random Forest...")

    rf_results = evaluate_model(
        "Random Forest",
        folds,
        get_rf_probabilities,
    )

    print("Evaluating Gradient Boosting...")

    gb_results = evaluate_model(
        "Gradient Boosting",
        folds,
        get_gb_probabilities,
    )

    rf_results = rf_results.sort_values(
        [
            "mean_f1",
            "worst_fold_f1",
        ],
        ascending=False,
    )

    gb_results = gb_results.sort_values(
        [
            "mean_f1",
            "worst_fold_f1",
        ],
        ascending=False,
    )

    rf_results.to_csv(
        RESULTS_DIR
        / "random_forest_thresholds.csv",
        index=False,
    )

    gb_results.to_csv(
        RESULTS_DIR
        / "gradient_boosting_thresholds.csv",
        index=False,
    )

    columns = [
        "threshold",
        "fold_1_precision",
        "fold_1_recall",
        "fold_1_f1",
        "fold_2_precision",
        "fold_2_recall",
        "fold_2_f1",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "worst_fold_f1",
    ]

    print()
    print(
        "=== Top Random Forest Thresholds ==="
    )

    print(
        rf_results[columns]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== Top Gradient Boosting Thresholds ==="
    )

    print(
        gb_results[columns]
        .head(10)
        .to_string(
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
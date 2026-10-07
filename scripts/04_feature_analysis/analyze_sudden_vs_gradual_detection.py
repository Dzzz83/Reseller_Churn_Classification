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
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path("results/error_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    RESULTS_DIR
    / "fold2_sudden_vs_gradual_detection.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "fold2_sudden_vs_gradual_predictions.csv"
)


FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]


TRAIN_SNAPSHOTS = [
    "2012-07-01",
    "2012-10-01",
]

VALIDATION_SNAPSHOT = "2013-04-01"

SUDDEN_RECENCY_DAYS = 30

RF_THRESHOLD = 0.34
GB_THRESHOLD = 0.24

SEEDS = [
    42,
    78,
    88,
    1034,
    2026,
]


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
        [
            non_churn,
            churn_sampled,
        ],
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


def get_rf_probabilities(
    x_train,
    y_train,
    x_val,
):
    probabilities = []

    for seed in SEEDS:
        model = build_rf(seed)

        model.fit(
            x_train,
            y_train,
        )

        probabilities.append(
            model.predict_proba(
                x_val
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def get_gb_probabilities(
    x_train,
    y_train,
    x_val,
):
    probabilities = []

    for seed in SEEDS:
        (
            x_balanced,
            y_balanced,
        ) = random_oversample(
            x_train,
            y_train,
            seed,
        )

        model = build_gb(seed)

        model.fit(
            x_balanced,
            y_balanced,
        )

        probabilities.append(
            model.predict_proba(
                x_val
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def classify_churn_type(row):
    if row["churn"] != 1:
        return "non_churn"

    if (
        row["recency_days"]
        <= SUDDEN_RECENCY_DAYS
    ):
        return "sudden_churn"

    return "gradual_churn"


def compute_group_metrics(
    model_name,
    threshold,
    df,
    group_name,
):
    if group_name == "overall":
        group = df.copy()

    elif group_name == "sudden":
        group = df[
            df["recency_days"]
            <= SUDDEN_RECENCY_DAYS
        ].copy()

    elif group_name == "gradual":
        group = df[
            df["recency_days"]
            > SUDDEN_RECENCY_DAYS
        ].copy()

    else:
        raise ValueError(group_name)

    probability_column = (
        "rf_probability"
        if model_name == "Random Forest"
        else "gb_probability"
    )

    prediction_column = (
        "rf_predicted_churn"
        if model_name == "Random Forest"
        else "gb_predicted_churn"
    )

    y_true = group["churn"].astype(int)
    y_pred = group[
        prediction_column
    ].astype(int)

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    churners = tp + fn

    churn_probability = group.loc[
        y_true == 1,
        probability_column,
    ]

    return {
        "model":
            model_name,

        "threshold":
            threshold,

        "group":
            group_name,

        "n_resellers":
            len(group),

        "actual_churners":
            churners,

        "detected_churners":
            tp,

        "missed_churners":
            fn,

        "false_alarms":
            fp,

        "correct_non_churn":
            tn,

        "churn_recall":
            (
                tp / churners
                if churners > 0
                else np.nan
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "mean_churn_probability":
            churn_probability.mean(),

        "median_churn_probability":
            churn_probability.median(),

        "group_pr_auc":
            average_precision_score(
                y_true,
                group[
                    probability_column
                ],
            ),
    }


def main():
    df = pd.read_csv(
        DATA_PATH
    )

    df[
        "snapshot"
    ] = pd.to_datetime(
        df["snapshot"]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    train = df[
        df["snapshot"].isin(
            TRAIN_SNAPSHOTS
        )
    ].copy()

    val = df[
        df["snapshot"]
        == VALIDATION_SNAPSHOT
    ].copy()

    assert len(val) == 340
    assert val["churn"].sum() == 64

    sudden_churners = val[
        (val["churn"] == 1)
        & (
            val["recency_days"]
            <= SUDDEN_RECENCY_DAYS
        )
    ]

    gradual_churners = val[
        (val["churn"] == 1)
        & (
            val["recency_days"]
            > SUDDEN_RECENCY_DAYS
        )
    ]

    assert len(
        sudden_churners
    ) == 28

    assert len(
        gradual_churners
    ) == 36

    x_train = train[
        FEATURES
    ]

    y_train = train[
        "churn"
    ].astype(int)

    x_val = val[
        FEATURES
    ]

    print(
        "=== Fold-2 Sudden vs "
        "Gradual Churn Detection ==="
    )

    print(
        f"Sudden definition: "
        f"recency_days <= "
        f"{SUDDEN_RECENCY_DAYS}"
    )

    print(
        f"Gradual definition: "
        f"recency_days > "
        f"{SUDDEN_RECENCY_DAYS}"
    )

    print()
    print(
        "Training Random Forest..."
    )

    val[
        "rf_probability"
    ] = get_rf_probabilities(
        x_train,
        y_train,
        x_val,
    )

    val[
        "rf_predicted_churn"
    ] = (
        val["rf_probability"]
        >= RF_THRESHOLD
    ).astype(int)

    print(
        "Training Gradient Boosting..."
    )

    val[
        "gb_probability"
    ] = get_gb_probabilities(
        x_train,
        y_train,
        x_val,
    )

    val[
        "gb_predicted_churn"
    ] = (
        val["gb_probability"]
        >= GB_THRESHOLD
    ).astype(int)

    val[
        "churn_type"
    ] = val.apply(
        classify_churn_type,
        axis=1,
    )

    rows = []

    for (
        model_name,
        threshold,
    ) in [
        (
            "Random Forest",
            RF_THRESHOLD,
        ),
        (
            "Gradient Boosting",
            GB_THRESHOLD,
        ),
    ]:

        for group_name in [
            "overall",
            "sudden",
            "gradual",
        ]:
            rows.append(
                compute_group_metrics(
                    model_name,
                    threshold,
                    val,
                    group_name,
                )
            )

    results = pd.DataFrame(
        rows
    )

    print()
    print(
        "=== Detection Summary ==="
    )

    display_columns = [
        "model",
        "threshold",
        "group",
        "actual_churners",
        "detected_churners",
        "missed_churners",
        "churn_recall",
        "false_alarms",
        "precision",
        "mean_churn_probability",
        "group_pr_auc",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=== Direct Churner Comparison ==="
    )

    for model_name in [
        "Random Forest",
        "Gradient Boosting",
    ]:
        sudden = results[
            (
                results["model"]
                == model_name
            )
            & (
                results["group"]
                == "sudden"
            )
        ].iloc[0]

        gradual = results[
            (
                results["model"]
                == model_name
            )
            & (
                results["group"]
                == "gradual"
            )
        ].iloc[0]

        print()
        print(model_name)

        print(
            "  Sudden:  "
            f"{int(sudden['detected_churners'])}"
            f"/{int(sudden['actual_churners'])} "
            f"caught "
            f"({sudden['churn_recall']:.2%})"
        )

        print(
            "  Gradual: "
            f"{int(gradual['detected_churners'])}"
            f"/{int(gradual['actual_churners'])} "
            f"caught "
            f"({gradual['churn_recall']:.2%})"
        )

        if sudden[
            "churn_recall"
        ] > 0:
            ratio = (
                gradual["churn_recall"]
                / sudden["churn_recall"]
            )

            print(
                "  Gradual/sudden "
                f"recall ratio: "
                f"{ratio:.2f}x"
            )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    export_columns = [
        "StoreID",
        "snapshot",
        "churn",
        "churn_type",
        "recency_days",
        "rf_probability",
        "rf_predicted_churn",
        "gb_probability",
        "gb_predicted_churn",
    ]

    val[
        export_columns
    ].to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Saved:",
        PREDICTIONS_PATH,
    )

    print()
    print(
        "Thresholds are PROVISIONAL "
        "and used only for diagnosis."
    )

    print(
        "Final-test labels were NOT used."
    )


if __name__ == "__main__":
    main()

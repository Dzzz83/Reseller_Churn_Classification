from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path("results/error_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]


# These are for diagnosis only.
# They are NOT added to the model yet.
DIAGNOSTIC_FEATURES = [
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
]


TRAIN_SNAPSHOTS = [
    "2012-07-01",
    "2012-10-01",
]

VALIDATION_SNAPSHOT = "2013-04-01"

THRESHOLD = 0.24

SEEDS = [
    42,
    78,
    88,
    1034,
    2026,
]


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


def get_identifier_column(df):
    candidates = [
        "StoreID",
        "store_id",
        "reseller_id",
        "ResellerID",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def train_and_predict(
    x_train,
    y_train,
    x_val,
):
    probabilities = []

    for seed in SEEDS:

        x_balanced, y_balanced = (
            random_oversample(
                x_train,
                y_train,
                seed,
            )
        )

        model = (
            HistGradientBoostingClassifier(
                **GB_PARAMS,
                random_state=seed,
                early_stopping=False,
            )
        )

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


def classify_result(row):
    if row["churn"] == 1:
        if row["predicted_churn"] == 1:
            return "TP_detected_churn"
        return "FN_missed_churn"

    if row["predicted_churn"] == 1:
        return "FP_false_alarm"

    return "TN_correct_non_churn"


def compare_churn_groups(df):
    caught = df[
        df["result_type"]
        == "TP_detected_churn"
    ]

    missed = df[
        df["result_type"]
        == "FN_missed_churn"
    ]

    rows = []

    for feature in DIAGNOSTIC_FEATURES:

        if feature not in df.columns:
            continue

        caught_values = pd.to_numeric(
            caught[feature],
            errors="coerce",
        )

        missed_values = pd.to_numeric(
            missed[feature],
            errors="coerce",
        )

        caught_median = caught_values.median()
        missed_median = missed_values.median()

        rows.append(
            {
                "feature": feature,
                "detected_median":
                    caught_median,
                "missed_median":
                    missed_median,
                "difference_missed_minus_detected":
                    missed_median
                    - caught_median,
                "detected_missing":
                    caught_values.isna().mean(),
                "missed_missing":
                    missed_values.isna().mean(),
            }
        )

    result = pd.DataFrame(rows)

    result[
        "absolute_difference"
    ] = (
        result[
            "difference_missed_minus_detected"
        ].abs()
    )

    return result.sort_values(
        "absolute_difference",
        ascending=False,
    )


def main():
    df = pd.read_csv(DATA_PATH)

    df["snapshot"] = pd.to_datetime(
        df["snapshot"]
    ).dt.strftime("%Y-%m-%d")

    train = df[
        df["snapshot"].isin(
            TRAIN_SNAPSHOTS
        )
    ].copy()

    val = df[
        df["snapshot"]
        == VALIDATION_SNAPSHOT
    ].copy()

    x_train = train[MODEL_FEATURES]
    y_train = train["churn"].astype(int)

    x_val = val[MODEL_FEATURES]

    probabilities = train_and_predict(
        x_train,
        y_train,
        x_val,
    )

    val["churn_probability"] = (
        probabilities
    )

    val["predicted_churn"] = (
        val["churn_probability"]
        >= THRESHOLD
    ).astype(int)

    val["result_type"] = val.apply(
        classify_result,
        axis=1,
    )

    print(
        "=== Fold 2 Gradient Boosting "
        "Error Analysis ==="
    )
    print(
        f"Threshold: {THRESHOLD:.2f}"
    )
    print()

    counts = (
        val["result_type"]
        .value_counts()
    )

    for label in [
        "TP_detected_churn",
        "FN_missed_churn",
        "FP_false_alarm",
        "TN_correct_non_churn",
    ]:
        print(
            f"{label}: "
            f"{counts.get(label, 0)}"
        )

    detected = counts.get(
        "TP_detected_churn",
        0,
    )

    missed = counts.get(
        "FN_missed_churn",
        0,
    )

    print()
    print(
        "Actual churners:",
        detected + missed,
    )

    if detected + missed > 0:
        print(
            "Churn recall:",
            f"{detected / (detected + missed):.2%}",
        )

    print()
    print(
        "=== Probability by Group ==="
    )

    probability_summary = (
        val.groupby("result_type")[
            "churn_probability"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        probability_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison = compare_churn_groups(
        val
    )

    print()
    print(
        "=== Detected vs Missed "
        "Churners ==="
    )

    columns = [
        "feature",
        "detected_median",
        "missed_median",
        "difference_missed_minus_detected",
        "detected_missing",
        "missed_missing",
    ]

    print(
        comparison[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    id_column = get_identifier_column(
        val
    )

    export_columns = []

    if id_column is not None:
        export_columns.append(
            id_column
        )

    export_columns += [
        "snapshot",
        "churn",
        "churn_probability",
        "predicted_churn",
        "result_type",
    ]

    export_columns += [
        feature
        for feature
        in DIAGNOSTIC_FEATURES
        if feature in val.columns
    ]

    val[
        export_columns
    ].sort_values(
        "churn_probability",
        ascending=False,
    ).to_csv(
        RESULTS_DIR
        / "fold2_gb_predictions.csv",
        index=False,
    )

    val[
        val["result_type"]
        == "FN_missed_churn"
    ][export_columns].sort_values(
        "churn_probability",
        ascending=False,
    ).to_csv(
        RESULTS_DIR
        / "fold2_gb_missed_churners.csv",
        index=False,
    )

    val[
        val["result_type"]
        == "TP_detected_churn"
    ][export_columns].sort_values(
        "churn_probability",
        ascending=False,
    ).to_csv(
        RESULTS_DIR
        / "fold2_gb_detected_churners.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR
        / "fold2_gb_churn_feature_comparison.csv",
        index=False,
    )

    print()
    print(
        "Saved results to:",
        RESULTS_DIR,
    )

    print()
    print(
        "Final test labels were NOT used."
    )


if __name__ == "__main__":
    main()
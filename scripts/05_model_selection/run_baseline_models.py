from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from define_feature_sets import FEATURE_SETS
from test_preprocessing_pipelines import (
    build_tree_preprocessor,
    build_linear_basic_preprocessor,
    build_linear_log_preprocessor,
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


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "prevalence": y_true.mean(),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
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


def run_model(
    train,
    validation,
    features,
    model_name,
):
    X_train = train[features]
    y_train = train["churn"]

    X_val = validation[features]
    y_val = validation["churn"]

    if model_name == "logistic_basic":
        preprocessor = build_linear_basic_preprocessor(
            features
        )

        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

    elif model_name == "logistic_log":
        preprocessor = build_linear_log_preprocessor(
            features
        )

        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

    elif model_name == "random_forest":
        preprocessor = build_tree_preprocessor(
            features
        )

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(model_name)

    # Fit preprocessing on training only.
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    return evaluate(
        y_val.to_numpy(),
        probabilities,
    )


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 3.1: Baseline Models ===\n")

    model_names = [
        "logistic_basic",
        "logistic_log",
        "random_forest",
    ]

    results = []

    for fold_name, fold in FOLDS.items():
        train = df[
            df["snapshot"].isin(
                pd.to_datetime(fold["train"])
            )
        ]

        validation = df[
            df["snapshot"]
            == pd.Timestamp(fold["validation"])
        ]

        print(
            f"{fold_name}: "
            f"Train={len(train)}, "
            f"Validation={len(validation)}"
        )

        for feature_set_name, features in (
            FEATURE_SETS.items()
        ):
            for model_name in model_names:

                metrics = run_model(
                    train,
                    validation,
                    features,
                    model_name,
                )

                results.append({
                    "fold": fold_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    **metrics,
                })

                print(
                    f"  {feature_set_name:<8} "
                    f"{model_name:<17} "
                    f"PR-AUC={metrics['pr_auc']:.3f} "
                    f"F1={metrics['f1']:.3f}"
                )

    results = pd.DataFrame(results)

    print("\n=== Detailed Results ===")

    display_columns = [
        "fold",
        "feature_set",
        "model",
        "prevalence",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]

    print(
        results[display_columns]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Rolling-validation mean
    # --------------------------------------------------

    summary = (
        results
        .groupby(
            ["feature_set", "model"],
            as_index=False,
        )
        [
            [
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .mean()
    )

    print("\n=== Rolling Validation Mean ===")

    print(
        summary
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .round(3)
        .to_string(index=False)
    )

    best = summary.sort_values(
        "pr_auc",
        ascending=False,
    ).iloc[0]

    print("\nBest baseline by mean PR-AUC:")
    print(
        f"{best['model']} + "
        f"{best['feature_set']} "
        f"(PR-AUC={best['pr_auc']:.3f})"
    )

    print(
        "\n[PASS] Baseline evaluation completed"
    )
    print(
        "No imbalance methods, threshold tuning, "
        "or test data used."
    )


if __name__ == "__main__":
    main()
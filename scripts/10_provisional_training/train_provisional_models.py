from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

RESULTS_DIR = Path("results/provisional")
MODELS_DIR = Path("models/provisional")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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


FINAL_TRAIN_SNAPSHOTS = [
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
]


THRESHOLD = 0.5
RANDOM_STATE = 42


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Columns: {list(df.columns)}"
    )


def build_random_forest():
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
                    random_state=RANDOM_STATE,
                    class_weight=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting():
    return HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )


def model_builders():
    return {
        "random_forest": build_random_forest,
        "gradient_boosting": build_gradient_boosting,
    }


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
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
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
    }


def save_pr_curve(
    model_name,
    predictions_df,
):
    y_true = predictions_df["actual"].to_numpy()
    probabilities = predictions_df[
        "probability"
    ].to_numpy()

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.3f}",
    )

    prevalence = y_true.mean()
    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"No-skill = {prevalence:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_name.replace('_', ' ').title()} "
        "— Temporal Validation"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{model_name}_pr_curve.png",
        dpi=180,
    )

    plt.close()


def save_confusion_matrix(
    model_name,
    predictions_df,
):
    y_true = predictions_df["actual"].to_numpy()

    probabilities = predictions_df[
        "probability"
    ].to_numpy()

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-churn",
            "Churn",
        ],
    )

    display.plot()

    plt.title(
        f"{model_name.replace('_', ' ').title()}\n"
        f"Threshold = {THRESHOLD}"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / f"{model_name}_confusion_matrix.png",
        dpi=180,
    )

    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    snapshot_col = find_column(
        df,
        [
            "snapshot",
            "snapshot_date",
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

    df[snapshot_col] = pd.to_datetime(
        df[snapshot_col]
    ).dt.strftime("%Y-%m-%d")

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    print(
        "=== Provisional Churn Model Training ==="
    )
    print()
    print("Models:")
    print("  Random Forest")
    print("  Gradient Boosting")
    print()
    print(f"Features: {len(FEATURES)}")
    print("Primary metric: PR-AUC")
    print(
        f"Demo classification threshold: "
        f"{THRESHOLD}"
    )
    print()

    all_metrics = []
    all_predictions = []

    #
    # Temporal validation
    #
    for model_name, builder in model_builders().items():

        print(
            f"=== {model_name.replace('_', ' ').title()} ==="
        )

        model_predictions = []

        for fold_name, fold in FOLDS.items():

            train_df = df[
                df[snapshot_col].isin(
                    fold["train"]
                )
            ].copy()

            val_df = df[
                df[snapshot_col]
                == fold["validation"]
            ].copy()

            x_train = train_df[FEATURES]
            y_train = train_df[
                target_col
            ].astype(int)

            x_val = val_df[FEATURES]
            y_val = val_df[
                target_col
            ].astype(int)

            model = builder()

            model.fit(
                x_train,
                y_train,
            )

            probabilities = model.predict_proba(
                x_val
            )[:, 1]

            metrics = calculate_metrics(
                y_val,
                probabilities,
            )

            row = {
                "model": model_name,
                "fold": fold_name,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "validation_churn_rate": (
                    y_val.mean()
                ),
                **metrics,
            }

            all_metrics.append(row)

            fold_predictions = pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold_name,
                    "snapshot": fold[
                        "validation"
                    ],
                    "actual": y_val.to_numpy(),
                    "probability": probabilities,
                }
            )

            fold_predictions[
                "predicted"
            ] = (
                fold_predictions[
                    "probability"
                ] >= THRESHOLD
            ).astype(int)

            model_predictions.append(
                fold_predictions
            )

            print(
                f"{fold_name}: "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

        model_predictions = pd.concat(
            model_predictions,
            ignore_index=True,
        )

        all_predictions.append(
            model_predictions
        )

        pooled_metrics = calculate_metrics(
            model_predictions["actual"],
            model_predictions["probability"],
        )

        all_metrics.append(
            {
                "model": model_name,
                "fold": "pooled_oof",
                "train_rows": None,
                "validation_rows": len(
                    model_predictions
                ),
                "validation_churn_rate": (
                    model_predictions[
                        "actual"
                    ].mean()
                ),
                **pooled_metrics,
            }
        )

        save_pr_curve(
            model_name,
            model_predictions,
        )

        save_confusion_matrix(
            model_name,
            model_predictions,
        )

        print(
            f"Pooled OOF: "
            f"PR-AUC={pooled_metrics['pr_auc']:.4f}, "
            f"ROC-AUC={pooled_metrics['roc_auc']:.4f}"
        )

        print()

    #
    # Save validation outputs
    #
    metrics_df = pd.DataFrame(
        all_metrics
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        RESULTS_DIR
        / "temporal_validation_predictions.csv",
        index=False,
    )

    #
    # Mean fold PR-AUC comparison
    #
    fold_only = metrics_df[
        metrics_df["fold"] != "pooled_oof"
    ]

    comparison = (
        fold_only
        .groupby("model")
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            worst_fold_pr_auc=("pr_auc", "min"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
        .sort_values(
            "mean_pr_auc",
            ascending=False,
        )
    )

    comparison.to_csv(
        RESULTS_DIR
        / "model_comparison.csv",
        index=False,
    )

    print(
        "=== Temporal Validation Summary ==="
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    #
    # Provisional full historical training
    #
    print()
    print(
        "=== Training Provisional Full Models ==="
    )

    full_train = df[
        df[snapshot_col].isin(
            FINAL_TRAIN_SNAPSHOTS
        )
    ].copy()

    x_full = full_train[FEATURES]
    y_full = full_train[
        target_col
    ].astype(int)

    print(
        f"Historical observations: "
        f"{len(full_train)}"
    )
    print(
        f"Churn observations: "
        f"{y_full.sum()}"
    )
    print(
        f"Churn rate: "
        f"{y_full.mean():.4f}"
    )

    for model_name, builder in model_builders().items():

        model = builder()

        model.fit(
            x_full,
            y_full,
        )

        path = (
            MODELS_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            path,
        )

        print(
            f"Saved: {path}"
        )

    #
    # Save experiment metadata
    #
    metadata = pd.DataFrame(
        {
            "setting": [
                "status",
                "features",
                "threshold",
                "imbalance_strategy",
                "hyperparameter_tuning",
                "final_test_used",
            ],
            "value": [
                "PROVISIONAL / DEMO",
                ", ".join(FEATURES),
                str(THRESHOLD),
                "none",
                "none",
                "NO",
            ],
        }
    )

    metadata.to_csv(
        RESULTS_DIR
        / "experiment_metadata.csv",
        index=False,
    )

    print()
    print(
        "PROVISIONAL training complete."
    )
    print(
        "Final 2013-10 test labels were NOT used."
    )


if __name__ == "__main__":
    main()
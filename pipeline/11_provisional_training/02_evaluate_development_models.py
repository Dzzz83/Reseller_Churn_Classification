from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.model_settings import (
    MODEL_SEEDS,
    PROVISIONAL_THRESHOLDS,
    TUNED_GRADIENT_BOOSTING,
    TUNED_RANDOM_FOREST,
)
from reseller_churn.config.project_paths import RESULTS_DIR
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.modeling.seeded_predictions import (
    SeededProbabilityPredictor,
)
from reseller_churn.evaluation.metrics import (
    calculate_threshold_metrics,
)


OUTPUT_DIR = RESULTS_DIR / "provisional"


def evaluate_model_on_fold(
    model_name: str,
    prepared,
) -> tuple[dict[str, object], pd.DataFrame]:
    x_train = prepared.training_data[
        PRUNED_FEATURES
    ]
    y_train = prepared.training_data[
        "churn"
    ].astype(int)

    x_val = prepared.validation_data[
        PRUNED_FEATURES
    ]
    y_val = prepared.validation_data[
        "churn"
    ].astype(int)

    if model_name == "Random Forest":
        probabilities_by_seed = (
            SeededProbabilityPredictor
            .random_forest_by_seed(
                training_features=x_train,
                training_target=y_train,
                validation_features=x_val,
                settings=TUNED_RANDOM_FOREST,
                seeds=MODEL_SEEDS,
            )
        )

        threshold = PROVISIONAL_THRESHOLDS[
            "random_forest"
        ]

    elif model_name == "Gradient Boosting":
        probabilities_by_seed = (
            SeededProbabilityPredictor
            .gradient_boosting_by_seed(
                training_features=x_train,
                training_target=y_train,
                validation_features=x_val,
                settings=TUNED_GRADIENT_BOOSTING,
                seeds=MODEL_SEEDS,
                oversample_training=True,
            )
        )

        threshold = PROVISIONAL_THRESHOLDS[
            "gradient_boosting"
        ]

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    # Stage-08 ranking protocol:
    # score each seed independently, then average the scores.
    seed_pr_auc = [
        average_precision_score(
            y_val,
            probabilities,
        )
        for probabilities in probabilities_by_seed
    ]

    seed_roc_auc = [
        roc_auc_score(
            y_val,
            probabilities,
        )
        for probabilities in probabilities_by_seed
    ]

    # Threshold diagnostics use the ensemble-average probability.
    ensemble_probabilities = (
        SeededProbabilityPredictor.average(
            probabilities_by_seed
        )
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            target=y_val,
            probabilities=ensemble_probabilities,
            threshold=threshold,
        )
    )

    summary = {
        "model": model_name,
        "fold": prepared.name,
        "train_rows": len(prepared.training_data),
        "validation_rows": len(prepared.validation_data),
        "validation_churners": int(y_val.sum()),
        "validation_churn_rate": float(y_val.mean()),
        "mean_seed_pr_auc": float(
            np.mean(seed_pr_auc)
        ),
        "std_seed_pr_auc": float(
            np.std(seed_pr_auc, ddof=1)
        ),
        "mean_seed_roc_auc": float(
            np.mean(seed_roc_auc)
        ),
        "threshold": threshold,
        **threshold_metrics,
    }

    predictions = pd.DataFrame(
        {
            "model": model_name,
            "fold": prepared.name,
            "StoreID": (
                prepared.validation_data[
                    "StoreID"
                ].to_numpy()
            ),
            "snapshot": (
                prepared.validation_data[
                    "snapshot"
                ].to_numpy()
            ),
            "actual_churn": y_val.to_numpy(),
            "churn_probability":
                ensemble_probabilities,
        }
    )

    predictions[
        "predicted_churn"
    ] = (
        predictions[
            "churn_probability"
        ]
        >= threshold
    ).astype(int)

    return summary, predictions


def main() -> None:
    print(
        "=== 11.2 Development Model Evaluation ==="
    )
    print(
        "Final-test snapshot 2013-10-01 is NOT evaluated."
    )
    print()

    data = DatasetLoader.load_labeled_snapshots()
    temporal_data = TemporalDataset(data)

    summary_rows = []
    prediction_frames = []

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(
            fold
        )

        for model_name in (
            "Random Forest",
            "Gradient Boosting",
        ):
            summary, predictions = (
                evaluate_model_on_fold(
                    model_name=model_name,
                    prepared=prepared,
                )
            )

            summary_rows.append(summary)
            prediction_frames.append(
                predictions
            )

    summary = pd.DataFrame(
        summary_rows
    )

    overall = (
        summary.groupby("model")
        .agg(
            mean_pr_auc=(
                "mean_seed_pr_auc",
                "mean",
            ),
            worst_fold_pr_auc=(
                "mean_seed_pr_auc",
                "min",
            ),
            mean_roc_auc=(
                "mean_seed_roc_auc",
                "mean",
            ),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_f1=(
                "f1",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_pr_auc",
            ascending=False,
        )
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "development_fold_metrics.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "development_model_summary.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "development_predictions.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "=== Per-Fold Metrics ==="
    )

    print(
        summary[
            [
                "model",
                "fold",
                "mean_seed_pr_auc",
                "std_seed_pr_auc",
                "mean_seed_roc_auc",
                "threshold",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print(
        "=== Development Summary ==="
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print(
        "PR-AUC/ROC-AUC are ranking metrics."
    )
    print(
        "Precision/Recall/F1 use provisional "
        "development thresholds."
    )
    print(
        "Thresholds are NOT locked."
    )
    print(
        "Final-test labels were NOT used."
    )

    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {overall_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()

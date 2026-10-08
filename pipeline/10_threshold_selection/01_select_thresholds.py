from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.model_settings import (
    MODEL_SEEDS,
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


OUTPUT_DIR = RESULTS_DIR / "threshold_selection"
THRESHOLDS = np.arange(0.01, 1.00, 0.01)


def collect_fold_probabilities(
    temporal_data: TemporalDataset,
    model_name: str,
) -> dict[str, dict[str, object]]:
    predictions = {}

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(fold)

        x_train = prepared.training_data[
            PRUNED_FEATURES
        ]
        y_train = prepared.training_data[
            "churn"
        ].astype(int)
        x_val = prepared.validation_data[
            PRUNED_FEATURES
        ]

        if model_name == "Random Forest":
            by_seed = (
                SeededProbabilityPredictor
                .random_forest_by_seed(
                    training_features=x_train,
                    training_target=y_train,
                    validation_features=x_val,
                    settings=TUNED_RANDOM_FOREST,
                    seeds=MODEL_SEEDS,
                )
            )
        else:
            by_seed = (
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

        predictions[fold.name] = {
            "target":
                prepared.validation_data[
                    "churn"
                ].astype(int),
            "probabilities":
                SeededProbabilityPredictor.average(
                    by_seed
                ),
        }

    return predictions


def evaluate_thresholds(
    model_name: str,
    fold_predictions: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []

    for threshold in THRESHOLDS:
        row = {
            "model": model_name,
            "threshold": threshold,
        }

        fold_metrics = []

        for fold_name, prediction in fold_predictions.items():
            metrics = calculate_threshold_metrics(
                prediction["target"],
                prediction["probabilities"],
                threshold,
            )

            for metric_name, value in metrics.items():
                row[
                    f"{fold_name}_{metric_name}"
                ] = value

            fold_metrics.append(metrics)

        row["mean_precision"] = np.mean(
            [
                metrics["precision"]
                for metrics in fold_metrics
            ]
        )
        row["mean_recall"] = np.mean(
            [
                metrics["recall"]
                for metrics in fold_metrics
            ]
        )
        row["mean_f1"] = np.mean(
            [
                metrics["f1"]
                for metrics in fold_metrics
            ]
        )
        row["worst_fold_f1"] = np.min(
            [
                metrics["f1"]
                for metrics in fold_metrics
            ]
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "mean_f1",
            "worst_fold_f1",
        ],
        ascending=False,
    )


def main() -> None:
    print("=== 10. Threshold Selection ===")
    print("Threshold status: DEVELOPMENT / NOT LOCKED")

    temporal_data = TemporalDataset(
        DatasetLoader.load_labeled_snapshots()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, filename in [
        (
            "Random Forest",
            "random_forest_thresholds.csv",
        ),
        (
            "Gradient Boosting",
            "gradient_boosting_thresholds.csv",
        ),
    ]:
        print()
        print(f"Evaluating {model_name}...")

        results = evaluate_thresholds(
            model_name,
            collect_fold_probabilities(
                temporal_data,
                model_name,
            ),
        )

        results.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        print(
            results.head(10).to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    print()
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

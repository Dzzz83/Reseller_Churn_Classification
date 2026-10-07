from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_00_config.feature_sets import PRUNED_FEATURES
from reseller_churn.step_00_config.model_settings import (
    MODEL_SEEDS,
    PROVISIONAL_THRESHOLDS,
    TUNED_GRADIENT_BOOSTING,
    TUNED_RANDOM_FOREST,
)
from reseller_churn.step_00_config.project_paths import RESULTS_DIR
from reseller_churn.step_00_config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_01_data.temporal_dataset import TemporalDataset
from reseller_churn.step_03_modeling.seeded_predictions import (
    SeededProbabilityPredictor,
)
from reseller_churn.step_04_evaluation.metrics import (
    calculate_threshold_metrics,
)


RECENT_DAYS = 30
OUTPUT_DIR = RESULTS_DIR / "error_analysis"


def evaluate_group(
    model_name: str,
    target: pd.Series,
    probabilities,
    threshold: float,
) -> dict[str, object]:
    metrics = calculate_threshold_metrics(
        target,
        probabilities,
        threshold,
    )

    return {
        "model": model_name,
        "actual_churners": int(target.sum()),
        **metrics,
    }


def main() -> None:
    print("=== 05.3 Sudden vs Gradual Churn ===")

    data = DatasetLoader.load_labeled_snapshots()
    prepared = TemporalDataset(data).prepare_fold(
        DEVELOPMENT_FOLDS[1]
    )

    x_train = prepared.training_data[
        PRUNED_FEATURES
    ]
    y_train = prepared.training_data[
        "churn"
    ].astype(int)
    x_val = prepared.validation_data[
        PRUNED_FEATURES
    ]
    validation = prepared.validation_data.copy()

    rf_probabilities = (
        SeededProbabilityPredictor.average(
            SeededProbabilityPredictor.random_forest_by_seed(
                training_features=x_train,
                training_target=y_train,
                validation_features=x_val,
                settings=TUNED_RANDOM_FOREST,
                seeds=MODEL_SEEDS,
            )
        )
    )

    gb_probabilities = (
        SeededProbabilityPredictor.average(
            SeededProbabilityPredictor.gradient_boosting_by_seed(
                training_features=x_train,
                training_target=y_train,
                validation_features=x_val,
                settings=TUNED_GRADIENT_BOOSTING,
                seeds=MODEL_SEEDS,
                oversample_training=True,
            )
        )
    )

    validation["rf_probability"] = rf_probabilities
    validation["gb_probability"] = gb_probabilities

    sudden_mask = (
        validation["recency_days"]
        <= RECENT_DAYS
    )
    gradual_mask = ~sudden_mask

    rows = []

    for model_name, column, threshold in [
        (
            "Random Forest",
            "rf_probability",
            PROVISIONAL_THRESHOLDS["random_forest"],
        ),
        (
            "Gradient Boosting",
            "gb_probability",
            PROVISIONAL_THRESHOLDS["gradient_boosting"],
        ),
    ]:
        for group_name, mask in [
            ("sudden", sudden_mask),
            ("gradual", gradual_mask),
        ]:
            group = validation.loc[mask]

            metrics = evaluate_group(
                model_name=model_name,
                target=group["churn"].astype(int),
                probabilities=group[column].to_numpy(),
                threshold=threshold,
            )

            rows.append(
                {
                    **metrics,
                    "group": group_name,
                    "n_resellers": len(group),
                    "threshold": threshold,
                }
            )

    summary = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "03_sudden_vs_gradual_detection.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        summary[
            [
                "model",
                "group",
                "n_resellers",
                "actual_churners",
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
        "Sudden churn diagnostic: "
        f"recency_days <= {RECENT_DAYS}"
    )
    print("Thresholds are provisional.")
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

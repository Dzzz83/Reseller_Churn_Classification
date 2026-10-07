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
)
from reseller_churn.step_00_config.project_paths import RESULTS_DIR
from reseller_churn.step_00_config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_01_data.temporal_dataset import TemporalDataset
from reseller_churn.step_03_modeling.seeded_predictions import (
    SeededProbabilityPredictor,
)


OUTPUT_DIR = RESULTS_DIR / "error_analysis"


def classify_result(row: pd.Series) -> str:
    if row["churn"] == 1:
        return (
            "detected_churn"
            if row["predicted_churn"] == 1
            else "missed_churn"
        )

    return (
        "false_alarm"
        if row["predicted_churn"] == 1
        else "correct_non_churn"
    )


def main() -> None:
    print("=== 05.2 Fold-2 Error Analysis ===")

    data = DatasetLoader.load_labeled_snapshots()
    fold = DEVELOPMENT_FOLDS[1]

    prepared = TemporalDataset(
        data
    ).prepare_fold(fold)

    x_train = prepared.training_data[
        PRUNED_FEATURES
    ]
    y_train = prepared.training_data[
        "churn"
    ].astype(int)

    x_val = prepared.validation_data[
        PRUNED_FEATURES
    ]

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

    probabilities = (
        SeededProbabilityPredictor.average(
            probabilities_by_seed
        )
    )

    threshold = PROVISIONAL_THRESHOLDS[
        "gradient_boosting"
    ]

    result = prepared.validation_data.copy()
    result["churn_probability"] = probabilities
    result["predicted_churn"] = (
        result["churn_probability"]
        >= threshold
    ).astype(int)

    result["result_type"] = result.apply(
        classify_result,
        axis=1,
    )

    counts = (
        result["result_type"]
        .value_counts()
    )

    for name in [
        "detected_churn",
        "missed_churn",
        "false_alarm",
        "correct_non_churn",
    ]:
        print(
            f"{name}: "
            f"{int(counts.get(name, 0))}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "02_fold2_gradient_boosting_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(f"Threshold: {threshold:.2f} (provisional)")
    print(f"Saved: {output_path}")
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

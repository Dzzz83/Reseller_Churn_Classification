from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from reseller_churn.step_00_config.feature_sets import PRUNED_FEATURES
from reseller_churn.step_00_config.model_settings import (
    PROVISIONAL_THRESHOLDS,
    TUNED_GRADIENT_BOOSTING,
    TUNED_RANDOM_FOREST,
)
from reseller_churn.step_00_config.project_paths import (
    MODELS_DIR,
    RESULTS_DIR,
)
from reseller_churn.step_00_config.validation_settings import LABELED_SNAPSHOTS
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_03_modeling.model_factory import ModelFactory
from reseller_churn.step_03_modeling.resampling import (
    balance_classes_by_random_oversampling,
)


OUTPUT_MODELS_DIR = MODELS_DIR / "provisional"
OUTPUT_RESULTS_DIR = RESULTS_DIR / "provisional"


def main() -> None:
    print("=== 11. Provisional Model Training ===")
    print("Final-test labels: NOT USED")

    data = DatasetLoader.load_labeled_snapshots()

    training_data = data[
        data["snapshot"].isin(
            LABELED_SNAPSHOTS
        )
    ].copy()

    x_train = training_data[
        PRUNED_FEATURES
    ]
    y_train = training_data[
        "churn"
    ].astype(int)

    OUTPUT_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_forest = ModelFactory.create_random_forest(
        TUNED_RANDOM_FOREST,
        random_seed=42,
    )
    random_forest.fit(
        x_train,
        y_train,
    )

    balanced_x, balanced_y = (
        balance_classes_by_random_oversampling(
            features=x_train,
            target=y_train,
            random_seed=42,
        )
    )

    gradient_boosting = (
        ModelFactory.create_gradient_boosting(
            TUNED_GRADIENT_BOOSTING,
            random_seed=42,
        )
    )
    gradient_boosting.fit(
        balanced_x,
        balanced_y,
    )

    rf_path = (
        OUTPUT_MODELS_DIR
        / "random_forest.joblib"
    )
    gb_path = (
        OUTPUT_MODELS_DIR
        / "gradient_boosting.joblib"
    )

    joblib.dump(
        random_forest,
        rf_path,
    )
    joblib.dump(
        gradient_boosting,
        gb_path,
    )

    metadata = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "features": ", ".join(PRUNED_FEATURES),
                "imbalance_strategy": "none",
                "threshold_status": "provisional",
                "threshold": PROVISIONAL_THRESHOLDS[
                    "random_forest"
                ],
                "final_test_used": "NO",
            },
            {
                "model": "Gradient Boosting",
                "features": ", ".join(PRUNED_FEATURES),
                "imbalance_strategy": "random_oversampling",
                "threshold_status": "provisional",
                "threshold": PROVISIONAL_THRESHOLDS[
                    "gradient_boosting"
                ],
                "final_test_used": "NO",
            },
        ]
    )

    metadata.to_csv(
        OUTPUT_RESULTS_DIR
        / "experiment_metadata.csv",
        index=False,
    )

    print(
        f"Training observations: {len(training_data)}"
    )
    print(
        f"Churn observations: {int(y_train.sum())}"
    )
    print(f"Saved: {rf_path}")
    print(f"Saved: {gb_path}")
    print(
        "These models are PROVISIONAL because threshold "
        "selection is not frozen."
    )


if __name__ == "__main__":
    main()

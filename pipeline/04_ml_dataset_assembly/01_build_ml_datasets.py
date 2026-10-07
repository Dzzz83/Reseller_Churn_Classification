from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_00_config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
    FINAL_TEST_FEATURES_PATH,
    LABELED_SNAPSHOTS_PATH,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_01_data.ml_dataset_builder import MLDatasetBuilder


def main() -> None:
    print("=== 04. ML Dataset Assembly ===")

    engineered_features = pd.read_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    labeled_data, final_test_features = MLDatasetBuilder().build(
        orders=DatasetLoader.load_orders(),
        engineered_features=engineered_features,
    )

    labeled_data.to_csv(
        LABELED_SNAPSHOTS_PATH,
        index=False,
    )

    final_test_features.to_csv(
        FINAL_TEST_FEATURES_PATH,
        index=False,
    )

    print(
        f"Development observations: {len(labeled_data)}"
    )
    print(
        f"Development churners: {int(labeled_data['churn'].sum())}"
    )
    print(
        f"Final-test feature rows: {len(final_test_features)}"
    )
    print("Final-test labels: NOT ACCESSED")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reseller_churn.step_01_config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
    FINAL_TEST_FEATURES_PATH,
    LABELED_SNAPSHOTS_PATH,
)
from reseller_churn.step_02_data.dataset_loader import DatasetLoader
from reseller_churn.step_02_data.ml_dataset_builder import MLDatasetBuilder

import pandas as pd


def main() -> None:
    print("=== Step 02: Build ML Datasets ===")

    orders = DatasetLoader.load_orders()

    engineered_features = pd.read_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    builder = MLDatasetBuilder()

    labeled_data, final_test_features = builder.build(
        orders=orders,
        engineered_features=engineered_features,
    )

    LABELED_SNAPSHOTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        f"Labeled development observations: {len(labeled_data)}"
    )
    print(
        f"Development churners: {labeled_data['churn'].sum()}"
    )
    print(
        f"Final-test feature rows: {len(final_test_features)}"
    )
    print("Final-test labels: NOT ACCESSED")
    print()
    print(f"Saved: {LABELED_SNAPSHOTS_PATH}")
    print(f"Saved: {FINAL_TEST_FEATURES_PATH}")


if __name__ == "__main__":
    main()

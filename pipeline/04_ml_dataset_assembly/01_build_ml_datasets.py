from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.config.project_paths import (
    LEGACY_ENGINEERED_FEATURES_PATH,
    LEGACY_FINAL_TEST_FEATURES_PATH,
    LEGACY_LABELED_SNAPSHOTS_PATH,
)
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.ml_dataset_builder import build_ml_datasets


def main() -> None:
    print("=== 04. ML Dataset Assembly ===")

    engineered_features = pd.read_csv(
        LEGACY_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    labeled_data, final_test_features = build_ml_datasets(
        orders=DatasetLoader.load_orders(),
        engineered_features=engineered_features,
    )

    LEGACY_LABELED_SNAPSHOTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    labeled_data.to_csv(
        LEGACY_LABELED_SNAPSHOTS_PATH,
        index=False,
    )

    final_test_features.to_csv(
        LEGACY_FINAL_TEST_FEATURES_PATH,
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
    print(f"Saved generated data under: {LEGACY_LABELED_SNAPSHOTS_PATH.parent}")


if __name__ == "__main__":
    main()

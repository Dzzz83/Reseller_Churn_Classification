from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_01_config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
)
from reseller_churn.step_01_config.validation_settings import (
    ALL_FEATURE_SNAPSHOTS,
)
from reseller_churn.step_02_data.dataset_loader import DatasetLoader
from reseller_churn.step_03_features.feature_pipeline import FeaturePipeline
from reseller_churn.step_05_evaluation.regression_checks import (
    RegressionChecks,
)


def main() -> None:
    print("=== Step 03: Build All Features ===")

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    pipeline = FeaturePipeline()

    features = pipeline.build(
        orders=orders,
        stores=stores,
        snapshots=ALL_FEATURE_SNAPSHOTS,
    )

    if ALL_ENGINEERED_FEATURES_PATH.exists():
        reference = pd.read_csv(
            ALL_ENGINEERED_FEATURES_PATH,
            parse_dates=["snapshot"],
        )

        RegressionChecks.assert_feature_frames_match(
            expected=reference,
            actual=features,
        )

        print(
            "[PASS] Refactored features match the verified reference."
        )

    ALL_ENGINEERED_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        index=False,
    )

    print(f"Observations: {len(features)}")
    print(f"Features: {len(features.columns) - 2}")
    print(f"Saved: {ALL_ENGINEERED_FEATURES_PATH}")


if __name__ == "__main__":
    main()

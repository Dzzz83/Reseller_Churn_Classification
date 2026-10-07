from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_00_config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
)
from reseller_churn.step_00_config.validation_settings import (
    ALL_FEATURE_SNAPSHOTS,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_02_features.feature_pipeline import FeaturePipeline
from reseller_churn.step_04_evaluation.regression_checks import RegressionChecks


def main() -> None:
    print("=== 03. Feature Engineering ===")

    pipeline = FeaturePipeline()

    features = pipeline.build(
        orders=DatasetLoader.load_orders(),
        stores=DatasetLoader.load_stores(),
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

    features.to_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        index=False,
    )

    print(f"Rows: {len(features)}")
    print(f"Engineered features: {len(features.columns) - 2}")
    print(f"Saved: {ALL_ENGINEERED_FEATURES_PATH}")


if __name__ == "__main__":
    main()

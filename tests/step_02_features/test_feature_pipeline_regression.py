import pandas as pd

from reseller_churn.step_00_config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
)
from reseller_churn.step_00_config.validation_settings import (
    ALL_FEATURE_SNAPSHOTS,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_02_features.feature_pipeline import FeaturePipeline
from reseller_churn.step_04_evaluation.regression_checks import (
    RegressionChecks,
)


def test_refactored_features_match_reference() -> None:
    reference = pd.read_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    actual = FeaturePipeline().build(
        orders=DatasetLoader.load_orders(),
        stores=DatasetLoader.load_stores(),
        snapshots=ALL_FEATURE_SNAPSHOTS,
    )

    RegressionChecks.assert_feature_frames_match(
        expected=reference,
        actual=actual,
    )

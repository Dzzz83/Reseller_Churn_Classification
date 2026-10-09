import pandas as pd

from reseller_churn.config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
    FINAL_TEST_FEATURES_PATH,
    LABELED_SNAPSHOTS_PATH,
)
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.ml_dataset_builder import MLDatasetBuilder


def test_ml_dataset_assembly_matches_reference():
    """Dataset assembly must reproduce the verified datasets."""

    features = pd.read_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    actual_labeled, actual_final = MLDatasetBuilder().build(
        orders=DatasetLoader.load_orders(),
        engineered_features=features,
    )

    expected_labeled = pd.read_csv(
        LABELED_SNAPSHOTS_PATH,
        parse_dates=["snapshot"],
    )

    expected_final = pd.read_csv(
        FINAL_TEST_FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    for actual, expected in (
        (actual_labeled, expected_labeled),
        (actual_final, expected_final),
    ):
        assert list(actual.columns) == list(expected.columns)

        actual = actual.sort_values(
            ["snapshot", "StoreID"]
        ).reset_index(drop=True)

        expected = expected.sort_values(
            ["snapshot", "StoreID"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-10,
        )

    # The protected final-test data must remain unlabeled.
    assert "churn" not in actual_final.columns
    assert set(
        actual_final["snapshot"].dt.strftime("%Y-%m-%d")
    ) == {"2013-10-01"}

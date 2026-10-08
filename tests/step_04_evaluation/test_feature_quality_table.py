import pandas as pd

from reseller_churn.config.feature_sets import FULL_FEATURES
from reseller_churn.data.dataset_loader import DatasetLoader


def test_feature_quality_table_has_one_row_per_feature() -> None:
    data = DatasetLoader.load_labeled_snapshots()

    missingness = (
        data[FULL_FEATURES]
        .isna()
        .mean()
        .rename_axis("feature")
        .reset_index(name="missing_fraction")
    )

    correlations = (
        data[
            FULL_FEATURES + ["churn"]
        ]
        .corr(numeric_only=True)["churn"]
        .drop("churn")
        .rename_axis("feature")
        .reset_index(name="correlation_with_churn")
    )

    summary = missingness.merge(
        correlations,
        on="feature",
        validate="one_to_one",
    )

    assert len(summary) == len(FULL_FEATURES)
    assert summary["feature"].is_unique
    assert set(summary["feature"]) == set(FULL_FEATURES)
    assert pd.api.types.is_numeric_dtype(
        summary["missing_fraction"]
    )
    assert pd.api.types.is_numeric_dtype(
        summary["correlation_with_churn"]
    )

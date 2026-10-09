import pandas as pd
import pytest


from comparisons.validation_schedule_audit import (
    assess_schedule,
    inspect_partial_history_candidate,
)
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.features.feature_pipeline import FeaturePipeline


def test_actual_history_cannot_support_independent_pretest_holdout():
    bounds = assess_schedule("2011-05-31", "2013-10-01")

    assert bounds.earliest_training == pd.Timestamp("2012-05-31")
    assert bounds.earliest_temporal_validation == pd.Timestamp("2012-11-30")
    assert bounds.temporal_results_available == pd.Timestamp("2013-05-30")
    assert bounds.latest_development_holdout == pd.Timestamp("2013-04-01")
    assert not bounds.has_independent_holdout


def test_earlier_history_makes_a_shared_holdout_possible():
    bounds = assess_schedule("2010-05-31", "2013-10-01")

    assert bounds.has_independent_holdout


def test_holdout_on_day_temporal_labels_become_known_is_allowed():
    bounds = assess_schedule("2011-05-31", "2013-11-30")

    assert bounds.temporal_results_available == bounds.latest_development_holdout
    assert bounds.has_independent_holdout


@pytest.mark.parametrize("history_months,label_months", [(0, 6), (12, 0)])
def test_invalid_window_sizes_are_rejected(history_months, label_months):
    with pytest.raises(ValueError, match="positive"):
        assess_schedule(
            "2011-05-31",
            "2013-10-01",
            history_months=history_months,
            label_months=label_months,
        )


def test_relaxed_schedule_is_resolved_and_has_enough_labeled_rows():
    orders = DatasetLoader.load_orders()
    rows = inspect_partial_history_candidate(orders)

    assert rows == [
        ("2012-04-01", 200, 45),
        ("2012-10-01", 366, 47),
        ("2013-04-01", 340, 64),
    ]


def test_existing_feature_builder_handles_partial_history_without_redefinition():
    features = FeaturePipeline().build(
        DatasetLoader.load_orders(),
        DatasetLoader.load_stores(),
        ("2012-04-01", "2012-10-01", "2013-04-01"),
    )

    assert len(features) == 200 + 366 + 340
    assert features["snapshot"].nunique() == 3
    assert not features.duplicated(["StoreID", "snapshot"]).any()

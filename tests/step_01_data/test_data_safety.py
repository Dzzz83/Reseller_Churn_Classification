import pandas as pd
import pytest

from reseller_churn.config.validation_settings import (
    FINAL_TEST_SNAPSHOT,
    TemporalFold,
)
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.data.temporal_dataset import TemporalDataset


def test_churn_window_boundaries():
    """Orders at the exact boundaries must be classified correctly."""
    records = [
        (1, "2012-10-01"),  # Eligibility starts: included
        (1, "2013-04-01"),  # Label starts: included
        (2, "2013-03-31"),
        (2, "2013-09-30"),  # Last day inside label window
        (3, "2012-09-30"),  # Before eligibility: excluded
        (4, "2013-04-01"),  # No preceding activity: excluded
        (5, "2013-02-01"),  # Eligible, no future orders: churn
        (6, "2013-02-02"),
        (6, "2013-10-01"),  # Label endpoint: excluded
    ]

    orders = pd.DataFrame(
        [
            {
                "SalesOrderID": index,
                "StoreID": store,
                "OrderDate": pd.Timestamp(date),
            }
            for index, (store, date) in enumerate(records, start=1)
        ]
    )

    labels = build_churn_labels(
        orders,
        PredictionWindow.from_date("2013-04-01"),
    )

    actual = labels.set_index("StoreID")["churn"].to_dict()

    assert actual == {
        1: 0,
        2: 0,
        5: 1,
        6: 1,
    }


def make_temporal_data():
    return TemporalDataset(
        pd.DataFrame(
            {
                "snapshot": [
                    "2012-07-01",
                    "2012-10-01",
                    "2013-01-01",
                    "2013-04-01",
                    "2013-10-01",
                ],
                "StoreID": [1, 1, 1, 1, 1],
                "churn": [0, 0, 1, 0, 1],
            }
        )
    )


def test_valid_temporal_fold():
    fold = TemporalFold(
        name="valid_fold",
        training_snapshots=("2012-07-01", "2012-10-01"),
        validation_snapshot="2013-04-01",
    )

    prepared = make_temporal_data().prepare_fold(fold)

    assert len(prepared.training_data) == 2
    assert len(prepared.validation_data) == 1


def test_training_labels_must_be_available():
    fold = TemporalFold(
        name="invalid_fold",
        training_snapshots=("2013-01-01",),
        validation_snapshot="2013-04-01",
    )

    with pytest.raises(ValueError, match="unavailable"):
        make_temporal_data().prepare_fold(fold)


def test_missing_training_snapshot_is_rejected():
    fold = TemporalFold(
        name="missing_snapshot",
        training_snapshots=("2012-07-01", "2012-08-01"),
        validation_snapshot="2013-04-01",
    )

    with pytest.raises(ValueError, match="missing training snapshots"):
        make_temporal_data().prepare_fold(fold)


def test_protected_final_test_cannot_be_validation():
    fold = TemporalFold(
        name="protected_test",
        training_snapshots=("2012-07-01",),
        validation_snapshot=FINAL_TEST_SNAPSHOT,
    )

    with pytest.raises(ValueError, match="Protected final-test"):
        make_temporal_data().prepare_fold(fold)

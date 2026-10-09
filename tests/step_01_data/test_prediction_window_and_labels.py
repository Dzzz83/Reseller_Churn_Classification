import pandas as pd

from reseller_churn.config.validation_settings import (
    EXPECTED_LABELED_COUNTS,
)
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.prediction_window import PredictionWindow


def test_prediction_window_boundaries() -> None:
    window = PredictionWindow.from_date(
        "2013-04-01"
    )

    assert window.history_start == pd.Timestamp(
        "2012-04-01"
    )
    assert window.eligibility_start == pd.Timestamp(
        "2012-10-01"
    )
    assert window.label_start == pd.Timestamp(
        "2013-04-01"
    )
    assert window.label_end == pd.Timestamp(
        "2013-10-01"
    )


def test_verified_snapshot_counts() -> None:
    orders = DatasetLoader.load_orders()
    for snapshot, (
        expected_rows,
        expected_churners,
    ) in EXPECTED_LABELED_COUNTS.items():
        labels = build_churn_labels(
            orders,
            PredictionWindow.from_date(snapshot),
        )

        assert len(labels) == expected_rows
        assert int(labels["churn"].sum()) == expected_churners

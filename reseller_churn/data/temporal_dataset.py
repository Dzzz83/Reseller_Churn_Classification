from dataclasses import dataclass

import pandas as pd

from reseller_churn.config.validation_settings import TemporalFold
from reseller_churn.config.validation_settings import FINAL_TEST_SNAPSHOT
from reseller_churn.data.prediction_window import PredictionWindow

@dataclass(frozen=True)
class PreparedTemporalFold:
    name: str
    training_data: pd.DataFrame
    validation_data: pd.DataFrame


class TemporalDataset:
    """Select the rows belonging to one explicit past-to-future fold."""

    def __init__(
        self,
        labeled_snapshots: pd.DataFrame,
        snapshot_column: str = "snapshot",
    ) -> None:
        self.data = labeled_snapshots.copy()
        self.snapshot_column = snapshot_column
        self.data[self.snapshot_column] = pd.to_datetime(
            self.data[self.snapshot_column]
        ).dt.strftime("%Y-%m-%d")

    def prepare_fold(self, fold: TemporalFold) -> PreparedTemporalFold:
        validation_date = pd.Timestamp(fold.validation_snapshot)
        final_test_date = pd.Timestamp(FINAL_TEST_SNAPSHOT)

        # The protected final test must not be used for development.
        if validation_date >= final_test_date:
            raise ValueError(
                "Protected final-test snapshot cannot be used for validation."
            )

        # Every configured training snapshot must exist.
        available_snapshots = set(self.data[self.snapshot_column])

        missing = sorted(
            set(fold.training_snapshots) - available_snapshots
        )

        if missing:
            raise ValueError(
                f"{fold.name}: missing training snapshots: {missing}"
            )

        # All training labels must be available before validation.
        for snapshot in fold.training_snapshots:
            window = PredictionWindow.from_date(snapshot)

            if window.label_end > validation_date:
                raise ValueError(
                    f"{fold.name}: training label for {snapshot} "
                    f"is unavailable by {fold.validation_snapshot}."
                )

        training_data = self.data[
            self.data[self.snapshot_column].isin(
                fold.training_snapshots
            )
        ].copy()

        validation_data = self.data[
            self.data[self.snapshot_column]
            == fold.validation_snapshot
        ].copy()

        if training_data.empty or validation_data.empty:
            raise ValueError(
                f"{fold.name}: training or validation data is empty."
            )

        return PreparedTemporalFold(
            name=fold.name,
            training_data=training_data,
            validation_data=validation_data,
        )

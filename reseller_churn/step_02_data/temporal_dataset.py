from dataclasses import dataclass

import pandas as pd

from reseller_churn.step_01_config.validation_settings import TemporalFold


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

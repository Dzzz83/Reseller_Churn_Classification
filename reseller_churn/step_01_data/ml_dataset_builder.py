import pandas as pd

from reseller_churn.step_00_config.validation_settings import (
    EXPECTED_FINAL_TEST_ROWS,
    EXPECTED_LABELED_COUNTS,
    FINAL_TEST_SNAPSHOT,
    LABELED_SNAPSHOTS,
)
from reseller_churn.step_01_data.churn_labels import ChurnLabelBuilder
from reseller_churn.step_01_data.prediction_window import PredictionWindow


class MLDatasetBuilder:
    """Combine historical features with labels without exposing final-test labels."""

    def __init__(self) -> None:
        self.label_builder = ChurnLabelBuilder()

    def build(
        self,
        orders: pd.DataFrame,
        engineered_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = engineered_features.copy()
        features["snapshot"] = pd.to_datetime(
            features["snapshot"]
        )

        labeled_parts = []

        for snapshot_text in LABELED_SNAPSHOTS:
            window = PredictionWindow.from_date(snapshot_text)

            if window.label_end > pd.Timestamp(FINAL_TEST_SNAPSHOT):
                raise AssertionError(
                    "A development label reaches beyond the final-test snapshot."
                )

            snapshot_features = features[
                features["snapshot"] == window.snapshot
            ].copy()

            labels = self.label_builder.build(
                orders,
                window,
            )

            expected_rows, expected_churners = (
                EXPECTED_LABELED_COUNTS[snapshot_text]
            )

            if len(snapshot_features) != expected_rows:
                raise AssertionError(
                    f"{snapshot_text}: unexpected feature-row count."
                )

            result = snapshot_features.merge(
                labels[["StoreID", "churn"]],
                on="StoreID",
                how="left",
                validate="one_to_one",
            )

            if result["churn"].isna().any():
                raise AssertionError(
                    f"{snapshot_text}: missing churn labels."
                )

            result["churn"] = result["churn"].astype(int)

            if int(result["churn"].sum()) != expected_churners:
                raise AssertionError(
                    f"{snapshot_text}: unexpected churn count."
                )

            labeled_parts.append(result)

        labeled_data = pd.concat(
            labeled_parts,
            ignore_index=True,
        )

        final_test_features = features[
            features["snapshot"]
            == pd.Timestamp(FINAL_TEST_SNAPSHOT)
        ].copy()

        if len(final_test_features) != EXPECTED_FINAL_TEST_ROWS:
            raise AssertionError(
                "Unexpected final-test feature-row count."
            )

        if "churn" in final_test_features.columns:
            raise AssertionError(
                "Final-test features must not contain churn labels."
            )

        if not (
            labeled_data["snapshot"].max()
            < final_test_features["snapshot"].min()
        ):
            raise AssertionError(
                "Development snapshots overlap the final test."
            )

        return labeled_data, final_test_features

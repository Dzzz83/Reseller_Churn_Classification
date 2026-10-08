"""Explicitly different, individually valid reseller validation strategies.

GroupKFold measures unseen-reseller generalization within a mixed period.
Temporal folds measure prediction on later snapshots using labels that
would already be known at the time of prediction.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROTECTED_TEST_SNAPSHOT = pd.Timestamp("2013-10-01")

DEVELOPMENT_SNAPSHOTS = (
    "2012-05-01",
    "2012-08-01",
    "2012-11-01",
    "2013-02-01",
)

TEMPORAL_FOLD_DATES = (
    ("fold_1", ("2012-05-01",), "2012-11-01"),
    ("fold_2", ("2012-05-01", "2012-08-01"), "2013-02-01"),
)


@dataclass(frozen=True)
class ValidationFold:
    name: str
    training_indices: np.ndarray
    validation_indices: np.ndarray


class ValidationPlans:
    """Build auditable folds without examining churn labels."""

    @staticmethod
    def verify_development_period(frame: pd.DataFrame) -> None:
        allowed = set(DEVELOPMENT_SNAPSHOTS)
        present = set(frame["snapshot"].unique())
        if present != allowed:
            raise AssertionError(
                f"Unexpected development snapshots: {sorted(present)}"
            )

        if frame.duplicated(["StoreID", "snapshot"]).any():
            raise AssertionError("Duplicate reseller/snapshot keys")

        for snapshot in present:
            label_end = (
                pd.Timestamp(snapshot) + pd.DateOffset(months=6)
            )
            if label_end > PROTECTED_TEST_SNAPSHOT:
                raise AssertionError(
                    f"{snapshot} has labels crossing protected test date"
                )

    @staticmethod
    def group_kfold(frame: pd.DataFrame) -> list[ValidationFold]:
        ValidationPlans.verify_development_period(frame)

        splitter = GroupKFold(n_splits=5)
        folds = []

        for i, (training, validation) in enumerate(
            splitter.split(frame, groups=frame["StoreID"]),
            start=1,
        ):
            training_stores = set(frame.iloc[training]["StoreID"])
            validation_stores = set(frame.iloc[validation]["StoreID"])

            if training_stores & validation_stores:
                raise AssertionError(
                    "GroupKFold reused StoreID across train and validation"
                )

            folds.append(
                ValidationFold(
                    name=f"group_fold_{i}",
                    training_indices=training,
                    validation_indices=validation,
                )
            )

        all_validation_indices = np.concatenate(
            [fold.validation_indices for fold in folds]
        )
        if sorted(all_validation_indices.tolist()) != list(
            range(len(frame))
        ):
            raise AssertionError(
                "GroupKFold must validate each observation exactly once"
            )
        return folds

    @staticmethod
    def temporal(frame: pd.DataFrame) -> list[ValidationFold]:
        ValidationPlans.verify_development_period(frame)

        folds = []
        for fold_name, train_dates, validation_date in TEMPORAL_FOLD_DATES:
            validation_start = pd.Timestamp(validation_date)

            for date in train_dates:
                if (
                    pd.Timestamp(date) + pd.DateOffset(months=6)
                    > validation_start
                ):
                    raise AssertionError(
                        "Training label unavailable at validation date"
                    )

            training = np.flatnonzero(
                frame["snapshot"].isin(train_dates).to_numpy()
            )
            validation = np.flatnonzero(
                (frame["snapshot"] == validation_date).to_numpy()
            )
            if not len(training) or not len(validation):
                raise AssertionError(
                    f"{fold_name} has an empty training or validation set"
                )

            if not (
                frame.iloc[training]["snapshot"].max()
                < frame.iloc[validation]["snapshot"].min()
            ):
                raise AssertionError(
                    "Temporal fold violates chronology"
                )

            folds.append(
                ValidationFold(
                    name=fold_name,
                    training_indices=training,
                    validation_indices=validation,
                )
            )

        return folds

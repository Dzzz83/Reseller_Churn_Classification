from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from reseller_churn.step_01_config.validation_settings import TemporalFold
from reseller_churn.step_02_data.temporal_dataset import TemporalDataset


class TemporalEvaluator:
    """Evaluate models on explicit past-to-future folds."""

    def __init__(
        self,
        labeled_snapshots: pd.DataFrame,
        folds: tuple[TemporalFold, ...],
        target_column: str = "churn",
    ) -> None:
        self.dataset = TemporalDataset(
            labeled_snapshots
        )
        self.folds = folds
        self.target_column = target_column

    def prepare_model_data(
        self,
        fold: TemporalFold,
        feature_names: list[str],
    ) -> dict[str, object]:
        prepared = self.dataset.prepare_fold(
            fold
        )

        return {
            "training_features":
                prepared.training_data[
                    feature_names
                ],
            "training_target":
                prepared.training_data[
                    self.target_column
                ].astype(int),
            "validation_features":
                prepared.validation_data[
                    feature_names
                ],
            "validation_target":
                prepared.validation_data[
                    self.target_column
                ].astype(int),
            "validation_rows":
                prepared.validation_data,
        }

    @staticmethod
    def average_seed_probabilities(
        build_and_fit: Callable[
            [int],
            np.ndarray,
        ],
        seeds: tuple[int, ...],
    ) -> np.ndarray:
        probabilities = [
            build_and_fit(seed)
            for seed in seeds
        ]

        return np.mean(
            probabilities,
            axis=0,
        )

    @staticmethod
    def mean_seed_pr_auc(
        validation_target: pd.Series,
        predict_for_seed: Callable[
            [int],
            np.ndarray,
        ],
        seeds: tuple[int, ...],
    ) -> float:
        scores = [
            average_precision_score(
                validation_target,
                predict_for_seed(seed),
            )
            for seed in seeds
        ]

        return float(
            np.mean(scores)
        )

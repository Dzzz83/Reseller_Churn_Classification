import numpy as np
import pandas as pd

from reseller_churn.step_00_config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)
from reseller_churn.step_03_modeling.model_factory import ModelFactory
from reseller_churn.step_03_modeling.resampling import (
    balance_classes_by_random_oversampling,
)


class SeededProbabilityPredictor:
    """Run the same model configuration across explicit random seeds."""

    @staticmethod
    def random_forest_by_seed(
        training_features: pd.DataFrame,
        training_target: pd.Series,
        validation_features: pd.DataFrame,
        settings: RandomForestSettings,
        seeds: tuple[int, ...],
    ) -> list[np.ndarray]:
        probabilities = []

        for seed in seeds:
            model = ModelFactory.create_random_forest(
                settings=settings,
                random_seed=seed,
            )

            model.fit(
                training_features,
                training_target,
            )

            probabilities.append(
                model.predict_proba(
                    validation_features
                )[:, 1]
            )

        return probabilities

    @staticmethod
    def gradient_boosting_by_seed(
        training_features: pd.DataFrame,
        training_target: pd.Series,
        validation_features: pd.DataFrame,
        settings: GradientBoostingSettings,
        seeds: tuple[int, ...],
        oversample_training: bool,
    ) -> list[np.ndarray]:
        probabilities = []

        for seed in seeds:
            current_features = training_features
            current_target = training_target

            if oversample_training:
                (
                    current_features,
                    current_target,
                ) = balance_classes_by_random_oversampling(
                    features=training_features,
                    target=training_target,
                    random_seed=seed,
                )

            model = ModelFactory.create_gradient_boosting(
                settings=settings,
                random_seed=seed,
            )

            model.fit(
                current_features,
                current_target,
            )

            probabilities.append(
                model.predict_proba(
                    validation_features
                )[:, 1]
            )

        return probabilities

    @staticmethod
    def average(
        probabilities_by_seed: list[np.ndarray],
    ) -> np.ndarray:
        return np.mean(
            probabilities_by_seed,
            axis=0,
        )

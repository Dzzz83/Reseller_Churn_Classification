import numpy as np
import pandas as pd

from reseller_churn.config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.resampling import (
    balance_classes_by_random_oversampling,
)


def _predict_probabilities_by_seed(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    seeds: tuple[int, ...],
    create_model,
    oversample_training: bool = False,
) -> list[np.ndarray]:
    """Fit independent seeded models, changing only the training data if requested."""
    probabilities = []

    for seed in seeds:
        current_features, current_target = training_features, training_target
        if oversample_training:
            current_features, current_target = balance_classes_by_random_oversampling(
                features=training_features,
                target=training_target,
                random_seed=seed,
            )

        model = create_model(seed)
        model.fit(current_features, current_target)
        probabilities.append(model.predict_proba(validation_features)[:, 1])

    return probabilities


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
        return _predict_probabilities_by_seed(
            training_features,
            training_target,
            validation_features,
            seeds,
            create_model=lambda seed: ModelFactory.create_random_forest(
                settings=settings,
                random_seed=seed,
            ),
        )

    @staticmethod
    def gradient_boosting_by_seed(
        training_features: pd.DataFrame,
        training_target: pd.Series,
        validation_features: pd.DataFrame,
        settings: GradientBoostingSettings,
        seeds: tuple[int, ...],
        oversample_training: bool,
    ) -> list[np.ndarray]:
        return _predict_probabilities_by_seed(
            training_features,
            training_target,
            validation_features,
            seeds,
            create_model=lambda seed: ModelFactory.create_gradient_boosting(
                settings=settings,
                random_seed=seed,
            ),
            oversample_training=oversample_training,
        )

    @staticmethod
    def average(
        probabilities_by_seed: list[np.ndarray],
    ) -> np.ndarray:
        return np.mean(
            probabilities_by_seed,
            axis=0,
        )

"""Seed-specific training behavior, without fitting actual expensive ensembles."""

import numpy as np
import pandas as pd

from reseller_churn.config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.seeded_predictions import SeededProbabilityPredictor


def test_random_forest_seed_order_and_probability_averaging(monkeypatch):
    trained = []

    class DummyModel:
        def __init__(self, seed):
            self.seed = seed

        def fit(self, features, target):
            trained.append((self.seed, len(features), len(target)))

        def predict_proba(self, features):
            p = self.seed / 10
            return np.tile([1 - p, p], (len(features), 1))

    monkeypatch.setattr(
        ModelFactory,
        "create_random_forest",
        lambda settings, random_seed: DummyModel(random_seed),
    )

    features = pd.DataFrame({"feature": [1, 2, 3]})
    labels = pd.Series([0, 1, 0])
    validation = pd.DataFrame({"feature": [5, 6]})

    predictions = SeededProbabilityPredictor.random_forest_by_seed(
        features,
        labels,
        validation,
        RandomForestSettings(n_estimators=10),
        seeds=(2, 4),
    )

    assert trained == [(2, 3, 3), (4, 3, 3)]
    np.testing.assert_allclose(predictions, [[0.2, 0.2], [0.4, 0.4]])
    np.testing.assert_allclose(
        SeededProbabilityPredictor.average(predictions),
        [0.3, 0.3],
    )


def test_gradient_boosting_oversamples_training_only(monkeypatch):
    fitted_sizes = []
    predicted_sizes = []

    class DummyModel:
        def fit(self, features, target):
            fitted_sizes.append((len(features), list(pd.Series(target).value_counts().sort_index())))

        def predict_proba(self, features):
            predicted_sizes.append(len(features))
            return np.tile([0.6, 0.4], (len(features), 1))

    monkeypatch.setattr(
        ModelFactory,
        "create_gradient_boosting",
        lambda settings, random_seed: DummyModel(),
    )

    predictions = SeededProbabilityPredictor.gradient_boosting_by_seed(
        training_features=pd.DataFrame({"feature": [1, 2, 3, 4]}),
        training_target=pd.Series([0, 0, 0, 1]),
        validation_features=pd.DataFrame({"feature": [10, 20]}),
        settings=GradientBoostingSettings(),
        seeds=(11, 12),
        oversample_training=True,
    )

    assert fitted_sizes == [(6, [3, 3]), (6, [3, 3])]
    assert predicted_sizes == [2, 2]
    np.testing.assert_allclose(predictions, [[0.4, 0.4], [0.4, 0.4]])

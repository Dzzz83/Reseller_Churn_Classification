from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reseller_churn.step_00_config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)


class ModelFactory:
    """Create project models from explicit, reusable settings."""

    @staticmethod
    def create_logistic_regression() -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=None,
                    ),
                ),
            ]
        )

    @staticmethod
    def create_random_forest(
        settings: RandomForestSettings,
        random_seed: int,
    ) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=settings.n_estimators,
                        max_depth=settings.max_depth,
                        min_samples_leaf=settings.min_samples_leaf,
                        max_features=settings.max_features,
                        random_state=random_seed,
                        class_weight=None,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    @staticmethod
    def create_gradient_boosting(
        settings: GradientBoostingSettings,
        random_seed: int,
    ) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=settings.learning_rate,
            max_iter=settings.max_iter,
            max_leaf_nodes=settings.max_leaf_nodes,
            min_samples_leaf=settings.min_samples_leaf,
            l2_regularization=settings.l2_regularization,
            random_state=random_seed,
            early_stopping=False,
        )

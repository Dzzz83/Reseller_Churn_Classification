from dataclasses import dataclass


@dataclass(frozen=True)
class RandomForestSettings:
    n_estimators: int
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"


@dataclass(frozen=True)
class GradientBoostingSettings:
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0


MODEL_SEEDS = (42, 78, 88, 1034, 2026)
ARCHITECTURE_COMPARISON_SEEDS = tuple(range(20))

BASELINE_RANDOM_FOREST = RandomForestSettings(
    n_estimators=300,
)

BASELINE_GRADIENT_BOOSTING = GradientBoostingSettings()

TUNED_RANDOM_FOREST = RandomForestSettings(
    n_estimators=600,
    max_depth=5,
    min_samples_leaf=1,
    max_features="sqrt",
)

TUNED_GRADIENT_BOOSTING = GradientBoostingSettings(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=7,
    min_samples_leaf=20,
    l2_regularization=0.0,
)

PROVISIONAL_THRESHOLDS = {
    "random_forest": 0.34,
    "gradient_boosting": 0.24,
}

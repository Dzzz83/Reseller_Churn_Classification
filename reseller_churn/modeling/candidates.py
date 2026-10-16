"""Small, explicit candidate configurations for development comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.model_settings import (
    BASELINE_GRADIENT_BOOSTING,
    BASELINE_RANDOM_FOREST,
    MODEL_SEEDS,
)


ARCHITECTURES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)
IMBALANCE_STRATEGIES = (
    "none",
    "balanced_weights",
    "random_oversampling",
)

@dataclass(frozen=True)
class Candidate:
    """One complete architecture, feature, imbalance, parameter and seed policy."""

    architecture: str
    feature_set: str
    imbalance_strategy: str
    parameters: dict[str, Any]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
        if self.imbalance_strategy not in IMBALANCE_STRATEGIES:
            raise ValueError(
                f"Unknown imbalance strategy: {self.imbalance_strategy}"
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique.")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("seeds must be non-negative integers.")
        expected_parameter_keys = {
            "logistic_regression": {"C", "max_iter"},
            "random_forest": {
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            },
            "hist_gradient_boosting": {
                "learning_rate",
                "max_iter",
                "max_leaf_nodes",
                "min_samples_leaf",
                "l2_regularization",
            },
        }[self.architecture]
        if set(self.parameters) != expected_parameter_keys:
            raise ValueError(
                f"Unexpected parameters for {self.architecture}: "
                f"{sorted(self.parameters)}"
            )

        def require_positive_integer(name: str) -> None:
            value = self.parameters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{self.architecture} {name} must be a positive integer.")

        def require_positive_number(name: str) -> None:
            value = self.parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{self.architecture} {name} must be positive.")

        if self.architecture == "logistic_regression":
            require_positive_number("C")
            require_positive_integer("max_iter")
        elif self.architecture == "random_forest":
            require_positive_integer("n_estimators")
            require_positive_integer("min_samples_leaf")
            if self.parameters["max_depth"] is not None:
                require_positive_integer("max_depth")
            max_features = self.parameters["max_features"]
            if not (
                isinstance(max_features, str)
                and max_features in {"sqrt", "log2"}
            ) and not (
                isinstance(max_features, (int, float))
                and not isinstance(max_features, bool)
                and math.isfinite(max_features)
                and 0 < max_features <= 1
            ):
                raise ValueError("Random Forest max_features is invalid.")
        else:
            require_positive_number("learning_rate")
            require_positive_integer("max_iter")
            require_positive_integer("max_leaf_nodes")
            require_positive_integer("min_samples_leaf")
            l2_regularization = self.parameters["l2_regularization"]
            if (
                isinstance(l2_regularization, bool)
                or not isinstance(l2_regularization, (int, float))
                or not math.isfinite(l2_regularization)
                or l2_regularization < 0
            ):
                raise ValueError("Gradient Boosting regularization cannot be negative.")

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(FEATURE_SETS[self.feature_set])

    @property
    def candidate_id(self) -> str:
        identity = {
            "architecture": self.architecture,
            "feature_set": self.feature_set,
            "imbalance_strategy": self.imbalance_strategy,
            "parameters": self.parameters,
            "seeds": list(self.seeds),
        }
        canonical = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        return (
            f"{self.architecture}__{self.feature_set}__"
            f"{self.imbalance_strategy}__{digest}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "architecture": self.architecture,
            "feature_set": self.feature_set,
            "features": list(self.features),
            "imbalance_strategy": self.imbalance_strategy,
            "parameters": self.parameters,
            "seeds": list(self.seeds),
            "seed_aggregation": "mean_predicted_probability",
        }


def _parameter_options(tuned: bool) -> dict[str, tuple[dict[str, Any], ...]]:
    random_forest = asdict(BASELINE_RANDOM_FOREST)
    gradient_boosting = asdict(BASELINE_GRADIENT_BOOSTING)

    options = {
        "logistic_regression": ({"C": 1.0, "max_iter": 2000},),
        "random_forest": (random_forest,),
        "hist_gradient_boosting": (gradient_boosting,),
    }

    if not tuned:
        return options

    options["logistic_regression"] = (
        {"C": 0.25, "max_iter": 2000},
        {"C": 1.0, "max_iter": 2000},
    )
    options["random_forest"] = (
        random_forest,
        {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 600,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    )
    options["hist_gradient_boosting"] = (
        gradient_boosting,
        {
            **gradient_boosting,
            "max_leaf_nodes": 7,
        },
    )
    return options


def build_candidates(
    *,
    tuned: bool = False,
    architectures: tuple[str, ...] | None = None,
    feature_sets: tuple[str, ...] | None = None,
    imbalance_strategies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> list[Candidate]:
    """Build the bounded Cartesian comparison, with optional explicit tuning."""
    selected_architectures = architectures or ARCHITECTURES
    selected_features = feature_sets or tuple(FEATURE_SETS)
    selected_strategies = imbalance_strategies or IMBALANCE_STRATEGIES

    unknown_architectures = set(selected_architectures) - set(ARCHITECTURES)
    unknown_features = set(selected_features) - set(FEATURE_SETS)
    unknown_strategies = set(selected_strategies) - set(IMBALANCE_STRATEGIES)
    if unknown_architectures or unknown_features or unknown_strategies:
        raise ValueError(
            "Unknown candidate choices: "
            f"architectures={sorted(unknown_architectures)}, "
            f"feature_sets={sorted(unknown_features)}, "
            f"imbalance_strategies={sorted(unknown_strategies)}."
        )

    if seeds is not None and (
        not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers.")

    parameter_options = _parameter_options(tuned)
    candidates = []
    for architecture in selected_architectures:
        architecture_seeds = seeds or MODEL_SEEDS
        for feature_set in selected_features:
            for strategy in selected_strategies:
                for parameters in parameter_options[architecture]:
                    candidates.append(
                        Candidate(
                            architecture=architecture,
                            feature_set=feature_set,
                            imbalance_strategy=strategy,
                            parameters=dict(parameters),
                            seeds=tuple(architecture_seeds),
                        )
                    )
    return candidates


def candidate_from_dict(value: dict[str, Any]) -> Candidate:
    """Reconstruct and validate a candidate serialized in a manifest."""
    required = {
        "candidate_id",
        "architecture",
        "feature_set",
        "features",
        "imbalance_strategy",
        "parameters",
        "seeds",
        "seed_aggregation",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Candidate manifest is missing fields: {sorted(missing)}")

    candidate = Candidate(
        architecture=value["architecture"],
        feature_set=value["feature_set"],
        imbalance_strategy=value["imbalance_strategy"],
        parameters=dict(value["parameters"]),
        seeds=tuple(value["seeds"]),
    )
    if list(candidate.features) != value["features"]:
        raise ValueError("Manifest features do not match its named feature set.")
    if candidate.candidate_id != value["candidate_id"]:
        raise ValueError("Manifest candidate ID does not match its configuration.")
    if value["seed_aggregation"] != "mean_predicted_probability":
        raise ValueError("Unsupported seed aggregation policy in manifest.")

    return candidate

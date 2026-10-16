"""Leakage-safe candidate fitting and selected predictor persistence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from reseller_churn.artifacts import file_sha256
from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)
from reseller_churn.config.project_paths import PROJECT_ROOT
from reseller_churn.config.validation_settings import (
    DEVELOPMENT_FOLDS,
    FINAL_TEST_SNAPSHOT,
    LABELED_SNAPSHOTS,
    TemporalFold,
)
from reseller_churn.evaluation.selection import SELECTION_RULE
from reseller_churn.modeling.candidates import Candidate, candidate_from_dict
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.resampling import (
    balance_classes_by_random_oversampling,
)

RESERVED_COLUMNS = {"StoreID", "snapshot", "churn"}


def validate_labeled_development_data(data: pd.DataFrame) -> pd.DataFrame:
    """Require exactly the approved labeled development snapshots."""
    required_columns = RESERVED_COLUMNS
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Labeled data is missing columns: {sorted(missing_columns)}")

    validated = data.copy()
    validated["snapshot"] = pd.to_datetime(validated["snapshot"]).dt.strftime(
        "%Y-%m-%d"
    )
    observed_snapshots = set(validated["snapshot"].unique())
    approved_snapshots = set(LABELED_SNAPSHOTS)
    if FINAL_TEST_SNAPSHOT in observed_snapshots:
        raise ValueError(
            f"Protected final-test snapshot {FINAL_TEST_SNAPSHOT} "
            "is not allowed in development."
        )
    outside_approved = observed_snapshots - approved_snapshots
    if outside_approved:
        raise ValueError(
            "Development data contains snapshots outside LABELED_SNAPSHOTS: "
            f"{sorted(outside_approved)}"
        )
    missing_approved = approved_snapshots - observed_snapshots
    if missing_approved:
        raise ValueError(
            "Development data must contain every approved labeled snapshot; "
            f"missing {sorted(missing_approved)}."
        )
    if validated[["StoreID", "snapshot"]].duplicated().any():
        raise ValueError("Development data has duplicate StoreID/snapshot rows.")
    if validated["churn"].isna().any() or not set(validated["churn"].unique()) <= {
        0,
        1,
    }:
        raise ValueError("Development target must contain only non-missing 0/1 labels.")

    for feature_set, features in FEATURE_SETS.items():
        reserved_features = RESERVED_COLUMNS.intersection(features)
        if reserved_features:
            raise ValueError(
                f"Feature set {feature_set} contains reserved columns: "
                f"{sorted(reserved_features)}"
            )
        absent_features = set(features) - set(validated.columns)
        if absent_features:
            raise ValueError(
                f"Development data is missing features for {feature_set}: "
                f"{sorted(absent_features)}"
            )
    return validated


def dataframe_sha256(data: pd.DataFrame) -> str:
    """Hash normalized labeled rows for synthetic and in-memory run callers."""
    ordered = data.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    payload = ordered.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repository_identity() -> dict[str, object]:
    """Record the checked-out revision and whether local files differ from it."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    changes = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "has_local_changes": bool(changes.stdout.strip())
        if changes.returncode == 0
        else None,
    }


def fold_manifest(folds: tuple[TemporalFold, ...] = DEVELOPMENT_FOLDS) -> list[dict]:
    return [
        {
            "name": fold.name,
            "training_snapshots": list(fold.training_snapshots),
            "validation_snapshot": fold.validation_snapshot,
        }
        for fold in folds
    ]


def _create_candidate_model(candidate: Candidate, seed: int):
    parameters = candidate.parameters
    if candidate.architecture == "logistic_regression":
        return ModelFactory.create_logistic_regression(
            c=float(parameters["C"]),
            max_iter=int(parameters["max_iter"]),
        )
    if candidate.architecture == "random_forest":
        return ModelFactory.create_random_forest(
            RandomForestSettings(**parameters),
            random_seed=seed,
        )
    if candidate.architecture == "hist_gradient_boosting":
        return ModelFactory.create_gradient_boosting(
            GradientBoostingSettings(**parameters),
            random_seed=seed,
        )
    raise ValueError(f"Unsupported architecture: {candidate.architecture}")


def _fit_one_model(
    candidate: Candidate,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    seed: int,
):
    fit_features = training_features.copy()
    fit_target = training_target.astype(int).copy()

    if candidate.imbalance_strategy == "random_oversampling":
        fit_features, fit_target = balance_classes_by_random_oversampling(
            features=fit_features,
            target=fit_target,
            random_seed=seed,
        )

    model = _create_candidate_model(candidate, seed)
    fit_kwargs: dict[str, Any] = {}
    if candidate.imbalance_strategy == "balanced_weights":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=fit_target,
        )
        fit_kwargs[
            "model__sample_weight"
            if candidate.architecture in {"logistic_regression", "random_forest"}
            else "sample_weight"
        ] = sample_weight

    model.fit(fit_features, fit_target, **fit_kwargs)
    return model


@dataclass
class FittedProbabilityEnsemble:
    """A candidate's saved seed ensemble; thresholds live in selection manifests."""

    candidate: dict[str, Any]
    models: tuple[Any, ...]
    data_sha256: str

    @property
    def candidate_id(self) -> str:
        return str(self.candidate["candidate_id"])

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.candidate["features"])

    def predict_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        missing_features = set(self.features) - set(features.columns)
        if missing_features:
            raise ValueError(
                f"Prediction data is missing features: {sorted(missing_features)}"
            )
        reserved_features = RESERVED_COLUMNS.intersection(self.features)
        if reserved_features:
            raise ValueError(
                f"Saved predictor contains reserved features: "
                f"{sorted(reserved_features)}"
            )

        selected_features = features.loc[:, list(self.features)]
        by_seed = [
            model.predict_proba(selected_features)[:, 1] for model in self.models
        ]
        return np.mean(by_seed, axis=0)

    def predict(self, features: pd.DataFrame, threshold: float) -> np.ndarray:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1.")
        return (self.predict_probabilities(features) >= threshold).astype(int)


def fit_candidate_ensemble(
    candidate: Candidate,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    data_sha256: str,
) -> FittedProbabilityEnsemble:
    """Fit the same seed policy used during validation, on training rows only."""
    candidate_config = candidate_from_dict(candidate.to_dict())
    selected_features = training_features.loc[:, list(candidate_config.features)]
    fitted_models = tuple(
        _fit_one_model(
            candidate_config,
            selected_features,
            training_target,
            seed,
        )
        for seed in candidate_config.seeds
    )
    return FittedProbabilityEnsemble(
        candidate=candidate_config.to_dict(),
        models=fitted_models,
        data_sha256=data_sha256,
    )


def fit_and_predict_candidate(
    candidate: Candidate,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Fit one candidate's seed ensemble and return averaged probabilities."""
    selected_features = training_features.loc[:, list(candidate.features)]
    selected_validation_features = validation_features.loc[:, list(candidate.features)]
    models = tuple(
        _fit_one_model(
            candidate,
            selected_features,
            training_target,
            seed,
        )
        for seed in candidate.seeds
    )
    probabilities = np.mean(
        [model.predict_proba(selected_validation_features)[:, 1] for model in models],
        axis=0,
    )
    return probabilities, models


def validate_selection_manifest(
    manifest: dict[str, Any],
    expected_data_sha256: str | None = None,
) -> Candidate:
    """Reject stale, ineligible, or mismatched configuration manifests."""
    if manifest.get("status") != "selected":
        raise ValueError("Only a selected manifest can be used for refitting.")
    candidate = candidate_from_dict(manifest.get("candidate", {}))
    threshold = manifest.get("threshold")
    precision_floor = manifest.get("minimum_precision")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("Selected manifest has no finite threshold.")
    if not 0 <= threshold <= 1:
        raise ValueError("Selected manifest threshold must be between 0 and 1.")
    if not isinstance(precision_floor, (int, float)) or not np.isfinite(
        precision_floor
    ):
        raise ValueError("Selected manifest has no finite precision floor.")
    if not 0 <= precision_floor <= 1:
        raise ValueError("Selected manifest precision floor must be between 0 and 1.")
    if manifest.get("selection_rule") != SELECTION_RULE:
        raise ValueError("Selected manifest uses an unsupported selection rule.")
    expected_policy = {
        "seed_aggregation": "mean_predicted_probability",
        "seeds": list(candidate.seeds),
        "threshold_comparison": ">=",
    }
    if manifest.get("prediction_policy") != expected_policy:
        raise ValueError("Selected manifest prediction policy is inconsistent.")
    if manifest.get("approved_training_snapshots") != list(LABELED_SNAPSHOTS):
        raise ValueError("Manifest training snapshots differ from the approved set.")
    if manifest.get("folds") != fold_manifest():
        raise ValueError("Manifest development folds differ from the configured folds.")
    recorded_data_hash = manifest.get("data_sha256")
    if not isinstance(recorded_data_hash, str) or len(recorded_data_hash) != 64:
        raise ValueError("Manifest does not contain a valid development data hash.")
    try:
        int(recorded_data_hash, 16)
    except ValueError as error:
        raise ValueError(
            "Manifest development data hash is not hexadecimal."
        ) from error
    if manifest.get("data_sha256") != expected_data_sha256 and (
        expected_data_sha256 is not None
    ):
        raise ValueError("Manifest data hash does not match the development data.")

    fold_metrics = manifest.get("selected_fold_metrics", {})
    required_fold_names = {fold.name for fold in DEVELOPMENT_FOLDS}
    if set(fold_metrics) != required_fold_names:
        raise ValueError(
            "Manifest does not contain metrics for every development fold."
        )
    for fold_name, metrics in fold_metrics.items():
        precision = metrics.get("precision")
        if not isinstance(precision, (int, float)) or not np.isfinite(precision):
            raise ValueError(f"Manifest has invalid precision for {fold_name}.")
        if precision < precision_floor:
            raise ValueError(
                f"Manifest precision for {fold_name} is below its required floor."
            )

    return candidate


def refit_selected_candidate(
    data: pd.DataFrame,
    manifest: dict[str, Any],
    expected_data_sha256: str | None = None,
) -> FittedProbabilityEnsemble:
    """Refit a selected candidate on only the full approved labeled data."""
    validated_data = validate_labeled_development_data(data)
    current_hash = dataframe_sha256(validated_data)
    if expected_data_sha256 is not None and expected_data_sha256 != current_hash:
        raise ValueError("Development data hash does not match the expected hash.")
    candidate = validate_selection_manifest(manifest, current_hash)
    training_features = validated_data.loc[:, list(candidate.features)]
    training_target = validated_data["churn"].astype(int)
    return fit_candidate_ensemble(
        candidate,
        training_features,
        training_target,
        current_hash,
    )


def load_selected_predictor(
    manifest_path: Path,
) -> tuple[dict[str, Any], FittedProbabilityEnsemble]:
    """Load a selected model bundle only when its manifest and checksum agree."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = validate_selection_manifest(manifest)
    artifact = manifest.get("model_bundle")
    if not isinstance(artifact, dict):
        raise ValueError("Selected manifest does not identify a model bundle.")
    relative_path = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    if not relative_path or not expected_sha256:
        raise ValueError("Selected manifest does not identify a model bundle.")

    artifact_path = (PROJECT_ROOT / relative_path).resolve()
    if not artifact_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Model bundle path must remain inside the project.")
    if file_sha256(artifact_path) != expected_sha256:
        raise ValueError("Model bundle checksum does not match its manifest.")

    predictor = joblib.load(artifact_path)
    if not isinstance(predictor, FittedProbabilityEnsemble):
        raise ValueError("Saved model bundle has an unexpected type.")
    if predictor.candidate_id != candidate.candidate_id:
        raise ValueError("Saved model candidate does not match the manifest.")
    if predictor.candidate != candidate.to_dict():
        raise ValueError("Saved model configuration differs from the manifest.")
    if predictor.data_sha256 != manifest["data_sha256"]:
        raise ValueError("Saved model data hash does not match the manifest.")
    if predictor.features != candidate.features:
        raise ValueError("Saved model features do not match the manifest.")
    if len(predictor.models) != len(candidate.seeds):
        raise ValueError("Saved model count does not match the manifest seed policy.")
    return manifest, predictor

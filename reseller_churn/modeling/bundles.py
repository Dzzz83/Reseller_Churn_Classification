"""Portable, checksummed model artifacts with an explicit runtime contract."""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from reseller_churn.artifacts import file_sha256, runtime_versions
from reseller_churn.config.frozen_model import (
    CUTOFF,
    FEATURES,
    MODEL_ID,
    PARAMETERS,
    RETAINED_FEATURES,
    SEEDS,
    TRAINING_SNAPSHOTS,
)
from reseller_churn.data.preparation import date
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.ensemble import apply_threshold, mean_probabilities

RETAINED_SHA256 = "6ee382620eeae80bb2fce68734813e0323b7620a56c4b73f867ecfdad66894ba"


def model_spec(*, features=FEATURES):
    return {"id": MODEL_ID, "features": list(features), "parameters": PARAMETERS}


def artifact_manifest(path, threshold, *, training=None):
    """Metadata for a newly trained development artifact, separate from the bundle."""
    apply_threshold([], threshold)
    return {
        "schema": "production-model-v1",
        "model_id": MODEL_ID,
        "model_sha256": file_sha256(path),
        "features": list(FEATURES),
        "parameters": PARAMETERS,
        "seeds": list(SEEDS),
        "runtime": runtime_versions(),
        "threshold": threshold,
        "threshold_comparison": ">=",
        "seed_aggregation": "mean_predicted_probability",
        "weighting": "equal_period_class_cells_fitting_partition_only",
        "training_snapshots": list(TRAINING_SNAPSHOTS),
        "labels_available_at": "2013-10-01",
        "training": training,
    }


@dataclass(frozen=True)
class LoadedEnsemble:
    models: tuple
    manifest: dict

    def probabilities(self, frame):
        if set(frame) != {"StoreID", "snapshot", *FEATURES}:
            raise ValueError(
                "Scoring accepts exactly keys and the ten features; no outcomes."
            )
        frame = validate_features(frame)
        if frame.snapshot.min() < date(self.manifest["labels_available_at"]):
            raise ValueError("Model training labels were unavailable at the snapshot.")
        estimator_inputs = frame.loc[:, list(FEATURES)].copy()
        # Only the trusted retained artifact uses the former final feature name.
        # The loader validates this exact positional correspondence before scoring.
        estimator_inputs.columns = self.manifest["features"]
        return mean_probabilities(self.models, estimator_inputs)


def load_model(path: Path, manifest_path: Path | None = None) -> LoadedEnsemble:
    """Load trusted local pickle data only after checksum and runtime validation.

    The packaged manifest identifies the retained artifact. Future artifacts need
    an explicit manifest produced by the development command.
    """
    path = Path(path)
    metadata = (
        Path(manifest_path).read_text()
        if manifest_path is not None
        else files("reseller_churn.config").joinpath("retained_model.json").read_text()
    )
    manifest = json.loads(metadata)
    if manifest.get("schema") != "production-model-v1":
        raise ValueError("Unsupported model manifest schema.")
    if manifest.get("model_sha256") != file_sha256(path):
        raise ValueError("Model checksum differs from the artifact manifest.")
    if manifest.get("runtime") != runtime_versions():
        raise ValueError("Model runtime versions differ from the saved environment.")
    expected_features = (
        RETAINED_FEATURES if manifest["model_sha256"] == RETAINED_SHA256 else FEATURES
    )
    expected = {
        "model_id": MODEL_ID,
        "features": list(expected_features),
        "parameters": PARAMETERS,
        "seeds": list(SEEDS),
        "threshold_comparison": ">=",
        "seed_aggregation": "mean_predicted_probability",
        "weighting": "equal_period_class_cells_fitting_partition_only",
        "training_snapshots": list(TRAINING_SNAPSHOTS),
        "labels_available_at": "2013-10-01",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported model feature, training or aggregation contract.")
    apply_threshold([], manifest.get("threshold"))
    if manifest["model_sha256"] == RETAINED_SHA256 and manifest["threshold"] != CUTOFF:
        raise ValueError("The retained artifact requires its exact frozen cutoff.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"spec", "models"}:
        raise ValueError("Expected a bundle containing spec and models.")
    spec = bundle["spec"]
    if not isinstance(spec, dict) or any(
        spec.get(k) != v for k, v in model_spec(features=expected_features).items()
    ):
        raise ValueError("Unsupported model specification.")
    if set(spec) - {"id", "features", "parameters", "description"}:
        raise ValueError("Unsupported model specification fields.")
    models = tuple(bundle["models"])
    if len(models) != len(SEEDS):
        raise ValueError("Expected exactly five estimators.")
    for model, seed in zip(models, SEEDS):
        if type(model) is not HistGradientBoostingClassifier:
            raise ValueError("Unsupported estimator type.")
        if (
            model.get_params()
            != HistGradientBoostingClassifier(
                **PARAMETERS, random_state=seed
            ).get_params()
        ):
            raise ValueError("Estimator parameters or seeds differ from the contract.")
        check_is_fitted(model)
        if list(getattr(model, "feature_names_in_", ())) != list(expected_features):
            raise ValueError("Estimator feature order differs from the contract.")
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Estimator classes must be [0, 1].")
    return LoadedEnsemble(models, manifest)

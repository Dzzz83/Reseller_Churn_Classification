"""Explicit adapters for grouped selections and the retained period/class model."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from reseller_churn.artifacts import file_sha256, source_identity
from reseller_churn.config import frozen_model
from reseller_churn.config.frozen_model import FEATURES, PARAMETERS, SEEDS
from reseller_churn.modeling.candidates import candidate_from_dict
from reseller_churn.modeling.development import (
    FittedProbabilityEnsemble,
)
from reseller_churn.modeling.ensemble import apply_threshold, mean_probabilities
from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
)
from reseller_churn.modeling.model_factory import ModelFactory


@dataclass(frozen=True)
class LoadedEnsemble:
    models: tuple
    features: tuple[str, ...]
    identity: str
    sha256: str
    labels_available_at: str
    training_source: dict | None = None

    def probabilities(self, frame):
        return mean_probabilities(self.models, frame.loc[:, list(self.features)])


def _check_runtime(manifest):
    if manifest.get("source", {}).get("versions") != source_identity()["versions"]:
        raise ValueError("Model runtime versions differ from the saved environment.")


def _verify_model_file(path, expected):
    if not expected or file_sha256(path) != expected:
        raise ValueError("Model bundle checksum does not match its manifest.")


def _check_feature_classes(model, features):
    if list(getattr(model, "feature_names_in_", ())) != list(features):
        raise ValueError(
            "Saved estimator feature order differs from its specification."
        )
    if not np.array_equal(getattr(model, "classes_", None), [0, 1]):
        raise ValueError("Saved estimator must have ordered binary classes [0, 1].")


def load_period_class_model(path: Path) -> LoadedEnsemble:
    """Read only this model and its manifest; never verify adjacent score artifacts.

    This adapter supports the original ten-feature research contract explicitly.
    It does not relax the source-identity checks of the historical fixed loader.
    """
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    expected = manifest.get("outputs", {}).get(path.name)
    _verify_model_file(path, expected)
    _check_runtime(manifest)
    recorded_contract = (
        manifest.get("source", {})
        .get("package_sha256", {})
        .get("config/frozen_model.py")
    )
    if recorded_contract != file_sha256(Path(frozen_model.__file__)):
        raise ValueError("Saved frozen feature/training configuration is unsupported.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"spec", "models"}:
        raise ValueError("Expected a retained research bundle with spec and models.")
    spec, models = bundle["spec"], tuple(bundle["models"])
    if (
        spec.get("id") != "period_class_weighted"
        or spec.get("features") != list(FEATURES)
        or spec.get("parameters") != PARAMETERS
        or len(models) != len(SEEDS)
    ):
        raise ValueError("Unsupported period/class model specification.")
    for model, seed in zip(models, SEEDS):
        if not isinstance(model, HistGradientBoostingClassifier):
            raise ValueError("Unexpected estimator type in period/class bundle.")
        if any(
            model.get_params().get(k) != v
            for k, v in {**PARAMETERS, "random_state": seed}.items()
        ):
            raise ValueError(
                "Saved estimator parameters differ from the specification."
            )
        _check_feature_classes(model, FEATURES)
    return LoadedEnsemble(
        models, FEATURES, spec["id"], expected, "2013-10-01", manifest["source"]
    )


def load_grouped_model(path: Path) -> tuple[LoadedEnsemble, dict]:
    path = Path(path)
    evidence_path = path.parent.parent / "manifest.json"
    evidence = json.loads(evidence_path.read_text())
    relative_selection = path.relative_to(evidence_path.parent).as_posix()
    if file_sha256(path) != evidence.get("evidence", {}).get(relative_selection):
        raise ValueError("Selection checksum does not match grouped run evidence.")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "grouped-selection-v1"
        or manifest.get("status") != "selected"
    ):
        raise ValueError("Expected a selected grouped-development manifest.")
    _check_runtime(manifest)
    candidate = candidate_from_dict(manifest["candidate"])
    policy = ThresholdPolicy(**manifest["policy"])
    plan = manifest["validation_plan"]
    expected_plan = GroupedValidationPlan(
        plan["outer_folds"],
        plan["inner_folds"],
        tuple(plan["repeat_seeds"]),
        plan["inner_seed"],
    )
    if expected_plan.to_dict() != plan or policy.to_dict() != manifest["policy"]:
        raise ValueError("Unsupported grouped validation or cutoff policy metadata.")
    if manifest.get("threshold_comparison") != ">=":
        raise ValueError("Unsupported threshold comparison in grouped selection.")
    apply_threshold([], manifest["threshold"])
    artifact = manifest["model_bundle"]
    model_path = (path.parent / artifact["path"]).resolve()
    _verify_model_file(model_path, artifact["sha256"])
    predictor = joblib.load(model_path)
    if (
        not isinstance(predictor, FittedProbabilityEnsemble)
        or predictor.candidate != candidate.to_dict()
        or predictor.data_sha256 != manifest["data_sha256"]
        or len(predictor.models) != len(candidate.seeds)
    ):
        raise ValueError("Saved grouped ensemble differs from its selection manifest.")
    for model, seed in zip(predictor.models, candidate.seeds):
        expected = ModelFactory.create_candidate(candidate, seed)
        if type(model) is not type(expected):
            raise ValueError("Unexpected estimator type in grouped bundle.")
        _check_feature_classes(model, candidate.features)
        actual_model = (
            model.named_steps["model"] if hasattr(model, "named_steps") else model
        )
        expected_model = (
            expected.named_steps["model"]
            if hasattr(expected, "named_steps")
            else expected
        )
        if actual_model.get_params() != expected_model.get_params():
            raise ValueError("Saved estimator parameters differ from its candidate.")
    return LoadedEnsemble(
        tuple(predictor.models),
        candidate.features,
        candidate.candidate_id,
        artifact["sha256"],
        manifest["labels_available_at"],
        manifest["source"],
    ), manifest

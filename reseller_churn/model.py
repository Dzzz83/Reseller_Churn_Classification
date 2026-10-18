"""Fixed ensemble training, manifest-checked loading, and ranked scoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from reseller_churn.artifacts import (
    PACKAGE_HASH_FORMAT,
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_manifest,
)
from reseller_churn.config.frozen_model import (
    FEATURES,
    PARAMETERS,
    SEEDS,
    TRAINING_SNAPSHOTS,
    model_configuration,
)
from reseller_churn.data.preparation import (
    KEYS,
    DateLike,
    date,
    require_columns,
)
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.ensemble import mean_probabilities


class ProbabilityEstimator(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FixedEnsemble:
    models: tuple[ProbabilityEstimator, ...]
    configuration: dict[str, object]

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        features = validate_features(frame).loc[:, list(FEATURES)]
        return mean_probabilities(self.models, features)


def train(data_path: Path, output: Path, prediction_date: DateLike) -> Path:
    if Path(output).exists():
        raise FileExistsError("Model output already exists; choose a new directory.")
    data = validate_features(pd.read_csv(data_path))
    require_columns(data, ["churn"], "training data")
    if not data.churn.isin([0, 1]).all() or data.churn.nunique() != 2:
        raise ValueError("Training labels must be binary and contain both classes.")
    if set(data.snapshot.dt.strftime("%Y-%m-%d")) != set(TRAINING_SNAPSHOTS):
        raise ValueError(
            "Training must contain exactly the approved development snapshots."
        )
    prediction_date = date(prediction_date)
    labels_available = max(data.snapshot.unique()) + pd.DateOffset(months=6)
    if labels_available > prediction_date:
        raise ValueError(
            "Training labels are unavailable by the declared prediction date."
        )
    # Canonical order makes training independent of CSV row ordering.
    data = data.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    weights = compute_sample_weight("balanced", data.churn)
    models = []
    for seed in SEEDS:
        model = HistGradientBoostingClassifier(**PARAMETERS, random_state=seed)
        model.fit(data[list(FEATURES)], data.churn.astype(int), sample_weight=weights)
        models.append(model)
    output = new_output_directory(output)
    joblib.dump(
        FixedEnsemble(tuple(models), model_configuration()), output / "ensemble.joblib"
    )
    write_manifest(
        output,
        {
            "stage": "train",
            "configuration": model_configuration(),
            "inputs": input_identity({"development": data_path}),
            "training_rows": len(data),
            "training_churners": int(data.churn.sum()),
            "labels_available_at": str(date(labels_available).date()),
            "prediction_date": str(prediction_date.date()),
        },
    )
    return output


def load_model(directory: Path) -> tuple[FixedEnsemble, dict]:
    manifest = verify_manifest(directory)
    if (
        manifest.get("stage") != "train"
        or manifest.get("configuration") != model_configuration()
    ):
        raise ValueError("Model configuration differs from the supported frozen model.")
    current_identity = source_identity()
    if manifest["source"].get("package_hash_format") != PACKAGE_HASH_FORMAT:
        raise ValueError(
            "Legacy model source identity format; use its recorded package/environment "
            "or refit explicitly. Existing manifests must not be rewritten."
        )
    if manifest["source"]["package_sha256"] != current_identity["package_sha256"]:
        raise ValueError(
            "Model package code has changed; use its recorded version or refit explicitly."
        )
    if manifest["source"]["versions"] != current_identity["versions"]:
        raise ValueError(
            "Model runtime versions differ from the saved training environment."
        )
    ensemble = joblib.load(Path(directory) / "ensemble.joblib")
    if (
        not isinstance(ensemble, FixedEnsemble)
        or ensemble.configuration != model_configuration()
        or len(ensemble.models) != len(SEEDS)
    ):
        raise ValueError("Unexpected model bundle or seed count.")
    for seed, model in zip(SEEDS, ensemble.models):
        if not isinstance(model, HistGradientBoostingClassifier):
            raise ValueError("Unexpected estimator type.")
        if any(
            model.get_params()[name] != value
            for name, value in {**PARAMETERS, "random_state": seed}.items()
        ):
            raise ValueError("Saved estimator parameters differ from the frozen model.")
        if list(model.feature_names_in_) != list(FEATURES):
            raise ValueError("Saved feature order differs from the frozen model.")
    return ensemble, manifest


def ranked_predictions(ensemble: FixedEnsemble, frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_features(frame)
    result = frame[[*KEYS, "n_orders_3m"]].copy()
    result["churn_probability"] = ensemble.probabilities(frame)
    result = result.sort_values(
        ["snapshot", "churn_probability", "StoreID"], ascending=[True, False, True]
    ).reset_index(drop=True)
    result["rank"] = result.groupby("snapshot").cumcount() + 1
    return result


def predict(model_directory: Path, features_path: Path, output: Path) -> Path:
    ensemble, manifest = load_model(model_directory)
    # Preserve the serialized binary float. The default CSV parser can move a
    # feature across a fitted histogram boundary and change an account's score.
    frame = validate_features(pd.read_csv(features_path, float_precision="round_trip"))
    if frame.snapshot.min() < date(manifest["prediction_date"]):
        raise ValueError("Cannot score before the model's declared prediction date.")
    predictions = ranked_predictions(ensemble, frame)
    output = new_output_directory(output)
    predictions.to_csv(output / "ranked_predictions.csv", index=False)
    write_manifest(
        output,
        {
            "stage": "predict",
            "inputs": input_identity(
                {
                    "features": features_path,
                    "model_manifest": Path(model_directory) / "manifest.json",
                }
            ),
            "model_configuration": manifest["configuration"],
            "snapshots": sorted(predictions.snapshot.dt.strftime("%Y-%m-%d").unique()),
            "rows": len(predictions),
            "outcomes_accessed": False,
        },
    )
    return output

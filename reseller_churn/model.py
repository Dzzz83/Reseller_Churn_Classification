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
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_manifest,
)
from reseller_churn.config import (
    DISCOUNT_FEATURES,
    FEATURES,
    KEYS,
    PARAMETERS,
    SEEDS,
    TRAINING_SNAPSHOTS,
    model_configuration,
)
from reseller_churn.data import (
    require_columns,
    require_unique_keys,
)
from reseller_churn.dates import DateLike, parse_calendar_date


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate account keys and feature ranges without filling missing discounts."""
    require_columns(frame, [*KEYS, *FEATURES], "features")
    validated_features = frame.copy()
    require_unique_keys(validated_features, KEYS, "features")
    if validated_features.empty:
        raise ValueError("Feature population is empty.")
    validated_features["snapshot"] = validated_features.snapshot.map(
        parse_calendar_date
    )
    require_unique_keys(validated_features, KEYS, "features")
    for feature_name in FEATURES:
        validated_features[feature_name] = pd.to_numeric(
            validated_features[feature_name], errors="raise"
        )
        values = validated_features[feature_name].to_numpy(dtype=float)
        if np.isinf(values).any() or (
            feature_name not in DISCOUNT_FEATURES and np.isnan(values).any()
        ):
            raise ValueError(f"Invalid feature values: {feature_name}")
    for feature_name in ["n_orders_3m", "recency_days"]:
        values = validated_features[feature_name].to_numpy()
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid counts/days: {feature_name}")
    for feature_name in ["revenue_3m", "revenue_12m"]:
        if (validated_features[feature_name] < 0).any():
            raise ValueError(f"Negative revenue: {feature_name}")
    for feature_name in [
        "share_bikes",
        "share_accessories",
        "share_clothing",
        "weighted_discount_3m",
        "discount_frequency_12m",
    ]:
        if not validated_features[feature_name].dropna().between(0, 1).all():
            raise ValueError(f"Feature outside [0, 1]: {feature_name}")
    if not validated_features.discount_change_3m.dropna().between(-1, 1).all():
        raise ValueError("Discount change outside [-1, 1].")
    return validated_features


class ProbabilityEstimator(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FixedEnsemble:
    models: tuple[ProbabilityEstimator, ...]
    configuration: dict[str, object]

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        features = validate_features(frame).loc[:, list(FEATURES)]
        # Class 1 is churn; average its probability across the registered models.
        churn_probabilities_by_model = [
            model.predict_proba(features)[:, 1] for model in self.models
        ]
        return np.mean(churn_probabilities_by_model, axis=0)


def _validate_training_labels(
    training_samples: pd.DataFrame, prediction_date: pd.Timestamp
) -> pd.Timestamp:
    """Require approved snapshots and labels observable by the prediction date."""
    require_columns(training_samples, ["churn"], "training data")
    if (
        not training_samples.churn.isin([0, 1]).all()
        or training_samples.churn.nunique() != 2
    ):
        raise ValueError("Training labels must be binary and contain both classes.")
    training_snapshot_dates = set(training_samples.snapshot.dt.strftime("%Y-%m-%d"))
    if training_snapshot_dates != set(TRAINING_SNAPSHOTS):
        raise ValueError(
            "Training must contain exactly the approved development snapshots."
        )
    labels_available_at = max(training_samples.snapshot.unique()) + pd.DateOffset(
        months=6
    )
    if labels_available_at > prediction_date:
        raise ValueError(
            "Training labels are unavailable by the declared prediction date."
        )
    return parse_calendar_date(labels_available_at)


def _fit_fixed_ensemble(training_samples: pd.DataFrame) -> FixedEnsemble:
    """Fit each registered seed using weights calculated from training labels only."""
    # Canonical order makes training independent of CSV row ordering.
    training_samples = training_samples.sort_values(
        ["snapshot", "StoreID"]
    ).reset_index(drop=True)
    training_features = training_samples[list(FEATURES)]
    churn_labels = training_samples.churn.astype(int)
    sample_weights = compute_sample_weight("balanced", training_samples.churn)
    trained_models = []
    for seed in SEEDS:
        model = HistGradientBoostingClassifier(**PARAMETERS, random_state=seed)
        model.fit(training_features, churn_labels, sample_weight=sample_weights)
        trained_models.append(model)
    return FixedEnsemble(tuple(trained_models), model_configuration())


def train(data_path: Path, output: Path, prediction_date: DateLike) -> Path:
    """Validate historical training samples, fit the fixed ensemble, and save a bundle."""
    if Path(output).exists():
        raise FileExistsError("Model output already exists; choose a new directory.")
    training_samples = validate_features(pd.read_csv(data_path))
    prediction_date = parse_calendar_date(prediction_date)
    labels_available_at = _validate_training_labels(training_samples, prediction_date)
    ensemble = _fit_fixed_ensemble(training_samples)

    output_directory = new_output_directory(output)
    joblib.dump(ensemble, output_directory / "ensemble.joblib")
    write_manifest(
        output_directory,
        {
            "stage": "train",
            "configuration": model_configuration(),
            "inputs": input_identity({"development": data_path}),
            "training_rows": len(training_samples),
            "training_churners": int(training_samples.churn.sum()),
            "labels_available_at": str(labels_available_at.date()),
            "prediction_date": str(prediction_date.date()),
        },
    )
    return output_directory


def load_model(directory: Path) -> tuple[FixedEnsemble, dict]:
    """Load a trusted bundle only after checking integrity and code/runtime identity."""
    manifest = verify_manifest(directory)
    if (
        manifest.get("stage") != "train"
        or manifest.get("configuration") != model_configuration()
    ):
        raise ValueError("Model configuration differs from the supported frozen model.")
    current_identity = source_identity()
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
    _validate_saved_estimators(ensemble)
    return ensemble, manifest


def _validate_saved_estimators(ensemble: FixedEnsemble) -> None:
    """Check fitted estimator types, registered parameters, and feature order."""
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


def ranked_predictions(ensemble: FixedEnsemble, frame: pd.DataFrame) -> pd.DataFrame:
    """Rank descending scores per snapshot, breaking ties by StoreID."""
    validated_features = validate_features(frame)
    predictions = validated_features[[*KEYS, "n_orders_3m"]].copy()
    predictions["churn_probability"] = ensemble.probabilities(validated_features)
    predictions = predictions.sort_values(
        ["snapshot", "churn_probability", "StoreID"], ascending=[True, False, True]
    ).reset_index(drop=True)
    predictions["rank"] = predictions.groupby("snapshot").cumcount() + 1
    return predictions


def predict(model_directory: Path, features_path: Path, output: Path) -> Path:
    """Score unlabeled features without reading outcomes and save ranked probabilities."""
    ensemble, manifest = load_model(model_directory)
    # Preserve the serialized binary float. The default CSV parser can move a
    # feature across a fitted histogram boundary and change an account's score.
    scoring_features = validate_features(
        pd.read_csv(features_path, float_precision="round_trip")
    )
    if scoring_features.snapshot.min() < parse_calendar_date(
        manifest["prediction_date"]
    ):
        raise ValueError("Cannot score before the model's declared prediction date.")
    predictions = ranked_predictions(ensemble, scoring_features)
    output_directory = new_output_directory(output)
    predictions.to_csv(output_directory / "ranked_predictions.csv", index=False)
    write_manifest(
        output_directory,
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
    return output_directory

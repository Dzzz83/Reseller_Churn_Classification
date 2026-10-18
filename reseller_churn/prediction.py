"""Outcome-free score and flag exports from existing, validated model bundles."""

import json
from pathlib import Path

import pandas as pd

from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.data.preparation import KEYS, date
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.bundles import load_grouped_model, load_period_class_model
from reseller_churn.modeling.ensemble import apply_threshold


def _load_scoring_features(path):
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "features":
        raise ValueError("Scoring requires a feature-stage manifest.")
    required = {
        "coverage_from",
        "coverage_through",
        "loaded_through",
        "rows",
        "snapshots",
        "outputs",
    }
    if not required.issubset(manifest):
        raise ValueError(
            "Feature manifest lacks required coverage or provenance fields."
        )
    if file_sha256(path) != manifest.get("outputs", {}).get(path.name):
        raise ValueError("Feature checksum does not match its manifest.")
    frame = pd.read_csv(path, float_precision="round_trip")
    if {"churn", "actual_churn", "test_churn", "future_orders"} & set(frame):
        raise ValueError("Scoring feature files must not contain outcomes.")
    frame = validate_features(frame)
    if manifest.get("rows") != len(frame) or sorted(
        manifest.get("snapshots", [])
    ) != sorted(frame.snapshot.dt.strftime("%Y-%m-%d").unique()):
        raise ValueError("Feature rows or snapshots differ from the manifest.")
    start = frame.snapshot.min() - pd.DateOffset(months=12)
    if (
        date(manifest["coverage_from"]) > start
        or date(manifest["coverage_through"]) < frame.snapshot.max()
    ):
        raise ValueError(
            "Feature manifest does not declare the required source coverage."
        )
    if (
        "loaded_through" not in manifest
        or date(manifest["loaded_through"]) < frame.snapshot.max()
    ):
        raise ValueError("Feature manifest lacks sufficient effective loaded coverage.")
    return frame, manifest_path


def export_predictions(
    model_path: Path, features_path: Path, output: Path, threshold: float
):
    """Apply a supplied cutoff to full-precision mean scores; no fitting or outcomes."""
    model_path, features_path = Path(model_path), Path(features_path)
    apply_threshold([], threshold)
    if Path(output).exists():
        raise FileExistsError(
            "Prediction output already exists; choose a new directory."
        )
    if model_path.suffix == ".json":
        ensemble, _ = load_grouped_model(model_path)
        model_manifest = model_path
    else:
        ensemble = load_period_class_model(model_path)
        model_manifest = model_path.parent / "manifest.json"
    if tuple(ensemble.features) != FEATURES:
        raise ValueError(
            "This source export supports the fixed ten-feature contract only."
        )
    frame, feature_manifest = _load_scoring_features(features_path)
    if frame.snapshot.min() < date(ensemble.labels_available_at):
        raise ValueError(
            "Model training labels were unavailable at the scoring snapshot."
        )
    result = frame[[*KEYS, "n_orders_3m"]].copy()
    result["churn_probability"] = ensemble.probabilities(frame)
    result["threshold"] = threshold
    result["predicted_churn"] = apply_threshold(result.churn_probability, threshold)
    result["model_id"] = ensemble.identity
    result = result.sort_values(
        ["snapshot", "churn_probability", "StoreID"], ascending=[True, False, True]
    ).reset_index(drop=True)
    result["rank"] = result.groupby("snapshot").cumcount() + 1
    output = new_output_directory(output)
    result.to_csv(output / "predictions.csv", index=False)
    write_manifest(
        output,
        {
            "stage": "prediction_export",
            "model_id": ensemble.identity,
            "model_sha256": ensemble.sha256,
            "training_source": ensemble.training_source,
            "threshold": threshold,
            "threshold_comparison": ">=",
            "seed_aggregation": "mean_predicted_probability",
            "inputs": input_identity(
                {
                    "model": model_path,
                    "model_manifest": model_manifest,
                    "features": features_path,
                    "feature_manifest": feature_manifest,
                }
            ),
            "rows": len(result),
            "snapshots": sorted(result.snapshot.dt.strftime("%Y-%m-%d").unique()),
            "models_fitted": 0,
            "outcomes_accessed": False,
        },
    )
    return output

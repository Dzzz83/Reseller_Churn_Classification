"""Outcome-free single-snapshot Power BI export from existing estimators."""

from pathlib import Path

import numpy as np

from reseller_churn.artifacts import new_output_directory, write_manifest
from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.data.preparation import build_features, date
from reseller_churn.data.source_contract import load_verified_sources
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.bundles import load_model
from reseller_churn.modeling.ensemble import apply_threshold

CSV_COLUMNS = ["StoreID", "Snapshot", *FEATURES, "Model score", "Predicted churn"]


def prediction_frame(ensemble, frame):
    """Keep the validated estimator inputs intact, including native missingness."""
    if set(frame) != {"StoreID", "snapshot", *FEATURES}:
        raise ValueError(
            "Scoring requires exactly keys and the ten features; no outcomes."
        )
    frame = validate_features(frame).sort_values("StoreID").reset_index(drop=True)
    if frame.snapshot.nunique() != 1 or frame.StoreID.duplicated().any():
        raise ValueError(
            "Export requires one snapshot and one row per eligible reseller."
        )
    scores = ensemble.probabilities(frame)
    flags = apply_threshold(scores, ensemble.manifest["threshold"])
    result = frame.rename(columns={"snapshot": "Snapshot"})
    result["StoreID"] = result.StoreID.astype("int64")
    result["Snapshot"] = result.Snapshot.dt.strftime("%Y-%m-%d")
    result["Model score"] = scores
    result["Predicted churn"] = np.where(flags, "Yes", "No")
    return result[CSV_COLUMNS]


def export_predictions(
    *,
    model_path,
    model_manifest,
    paths,
    source_manifest,
    coverage_from,
    coverage_through,
    snapshot,
    output,
):
    """No fitting, threshold selection or outcome access occurs on this path."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Prediction output already exists; choose a new directory."
        )
    snapshot = date(snapshot)
    ensemble = load_model(model_path, model_manifest)
    source, provenance = load_verified_sources(
        paths,
        source_manifest,
        coverage_from,
        coverage_through,
        snapshot,
    )
    frame = build_features(source, snapshot)
    result = prediction_frame(ensemble, frame)
    output = new_output_directory(output)
    # pandas' default float representation round-trips binary64 values. Do not
    # supply a display precision: flags were already computed at full precision.
    result.to_csv(output / "predictions.csv", index=False, na_rep="")
    write_manifest(
        output,
        {
            "stage": "power_bi_export",
            **provenance,
            "model": ensemble.manifest,
            "rows": len(result),
            "snapshot": str(snapshot.date()),
            "features": list(FEATURES),
            "estimator_features": ensemble.manifest["features"],
            "feature_name_mapping": {
                public: saved
                for public, saved in zip(FEATURES, ensemble.manifest["features"])
                if public != saved
            },
            "model_sha256": ensemble.manifest["model_sha256"],
            "threshold": ensemble.manifest["threshold"],
            "threshold_comparison": ">=",
            "seed_aggregation": "mean_predicted_probability",
            "models_fitted": 0,
            "outcomes_accessed": False,
        },
    )
    return output

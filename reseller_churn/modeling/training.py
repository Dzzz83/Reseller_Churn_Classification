"""One five-seed model, fitting-only weights and grouped development validation."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from reseller_churn.artifacts import (
    new_output_directory,
    write_json,
    write_manifest,
)
from reseller_churn.config.frozen_model import (
    FEATURES,
    PARAMETERS,
    SEEDS,
    TRAINING_SNAPSHOTS,
)
from reseller_churn.data.preparation import build_development
from reseller_churn.data.source_contract import load_verified_sources
from reseller_churn.evaluation import assessment_metrics
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.bundles import artifact_manifest, model_spec
from reseller_churn.modeling.ensemble import mean_probabilities
from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
    cutoff_scan,
)
from reseller_churn.modeling.weighting import period_weights


def validate_development(data):
    if set(data) != {"StoreID", "snapshot", *FEATURES, "churn"}:
        raise ValueError("Development requires exactly keys, ten features and churn.")
    result = validate_features(data)
    if set(result.snapshot.dt.strftime("%Y-%m-%d")) != set(TRAINING_SNAPSHOTS):
        raise ValueError("Development requires exactly the four approved snapshots.")
    period_weights(result)
    return result.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def fit_models(data):
    """Compute weights from this fitting partition only; no class_weight multiplier."""
    data = validate_development(data)
    weights = period_weights(data)
    return tuple(
        HistGradientBoostingClassifier(**PARAMETERS, random_state=seed).fit(
            data[list(FEATURES)],
            data.churn,
            sample_weight=weights,
        )
        for seed in SEEDS
    )


def oof_predictions(data, splits):
    values = np.full(len(data), np.nan)
    for training, held in splits:
        models = fit_models(data.iloc[training])
        values[held] = mean_probabilities(models, data.iloc[held][list(FEATURES)])
    if not np.isfinite(values).all():
        raise ValueError("Grouped OOF must cover every development row.")
    return values


def develop(data, output, *, plan=None, policy=None, provenance=None):
    """Assess cutoff selection on outer groups, then select using full inner OOF.

    This future-development command never changes the retained artifact. Outer
    outcomes assess the procedure; only inner OOF chooses each cutoff.
    """
    if Path(output).exists():
        raise FileExistsError("Development output already exists.")
    data = validate_development(data)
    plan = plan or GroupedValidationPlan()
    policy = policy or ThresholdPolicy()
    outer = list(plan.outer_splits(data))
    # Preflight every fitting cell before any expensive fits or output writes.
    full_inner = plan.inner_splits(data)
    inner_by_outer = []
    for _, _, training, _ in outer:
        train = data.iloc[training]
        period_weights(train)
        inner = plan.inner_splits(train)
        inner_by_outer.append(inner)
        for fitting, _ in inner:
            period_weights(train.iloc[fitting])
    for training, _ in full_inner:
        period_weights(data.iloc[training])
    assessments, selections = [], []
    for (repeat, fold, training, held), inner in zip(outer, inner_by_outer):
        train, valid = data.iloc[training], data.iloc[held]
        scores = oof_predictions(train, inner)
        chosen = policy.select(cutoff_scan(train.churn, scores))
        selections.append({"repeat": repeat, "fold": fold, "choice": chosen})
        if chosen is not None:
            probabilities = mean_probabilities(fit_models(train), valid[list(FEATURES)])
            assessments.append(
                valid[["StoreID", "snapshot", "churn"]].assign(
                    repeat=repeat,
                    fold=fold,
                    score=probabilities,
                    cutoff=chosen["cutoff"],
                    predicted_churn=probabilities >= chosen["cutoff"],
                )
            )
    scores = oof_predictions(data, full_inner)
    choice = policy.select(cutoff_scan(data.churn, scores))
    # No feasible cutoff is a recorded outcome, never a fallback threshold.
    output = new_output_directory(output)
    data[["StoreID", "snapshot", "churn"]].assign(score=scores).to_csv(
        output / "development_oof.csv",
        index=False,
    )
    if assessments:
        assessment = pd.concat(assessments, ignore_index=True)
        assessment.to_csv(output / "grouped_assessment.csv", index=False)
        assessment_metrics(assessment, len(data)).to_csv(
            output / "grouped_metrics.csv", index=False
        )
    if choice is not None:
        path = output / "models.joblib"
        joblib.dump({"spec": model_spec(), "models": fit_models(data)}, path)
        identity = pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()

        write_json(
            output / "model_manifest.json",
            artifact_manifest(
                path,
                choice["cutoff"],
                training={
                    "data_sha256": hashlib.sha256(identity).hexdigest(),
                    "rows": len(data),
                    "plan": plan.to_dict(),
                    "policy": policy.to_dict(),
                },
            ),
        )
    write_manifest(
        output,
        {
            "stage": "grouped_development",
            "plan": plan.to_dict(),
            "policy": policy.to_dict(),
            "choice": choice,
            "outer_selections": selections,
            "inputs": provenance or {},
            "status": "selected" if choice else "no_feasible_cutoff",
            "outcomes_scope": "approved_development_only",
        },
    )
    return output


def develop_from_sources(
    *, paths, source_manifest, coverage_from, coverage_through, output
):
    source, provenance = load_verified_sources(
        paths,
        source_manifest,
        coverage_from,
        coverage_through,
        "2013-10-01",
    )
    return develop(build_development(source), output, provenance=provenance)

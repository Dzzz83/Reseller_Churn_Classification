"""Nested grouped selection, outer assessment and full-development persistence."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reseller_churn.artifacts import (
    file_sha256,
    source_identity,
    write_json,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.config.validation_settings import LABELED_SNAPSHOTS
from reseller_churn.data.preparation import KEYS, date, require_columns, unique_ids
from reseller_churn.evaluation.metrics import calculate_threshold_metrics
from reseller_churn.features.validation import validate_features
from reseller_churn.modeling.development import (
    dataframe_sha256,
    fit_and_predict_candidate,
    fit_candidate_ensemble,
)
from reseller_churn.modeling.ensemble import apply_threshold
from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
    cutoff_scan,
)
from reseller_churn.modeling.weighting import period_weights


def validate_grouped_data(data, candidates):
    require_columns(data, [*KEYS, "churn"], "development")
    result = data.copy()
    result["snapshot"] = result.snapshot.map(date)
    unique_ids(result, KEYS, "development")
    if set(result.snapshot.dt.strftime("%Y-%m-%d")) != set(LABELED_SNAPSHOTS):
        raise ValueError(
            "Grouped development requires exactly the approved development snapshots."
        )
    if not result.churn.isin([0, 1]).all() or result.churn.nunique() != 2:
        raise ValueError("Grouped development requires both binary outcome classes.")
    if {"actual_churn", "test_churn", "future_orders"} & set(result):
        raise ValueError("Outcome-derived columns are not development inputs.")
    for candidate in candidates:
        require_columns(result, candidate.features, "development features")
        values = result[list(candidate.features)].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite development features are invalid.")
    if set(FEATURES).issubset(result):
        result = validate_features(result)
    return result.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def inner_oof_scores(data, candidates, plan):
    """Fit with training rows only and pass unlabeled feature matrices to prediction."""
    splits = plan.inner_splits(data)
    assignment = np.full(len(data), -1, dtype=int)
    for fold, (_, held) in enumerate(splits):
        assignment[held] = fold
    scores = {}
    for candidate in candidates:
        values = np.full(len(data), np.nan)
        for training, held in splits:
            train, valid = data.iloc[training], data.iloc[held]
            values[held], _ = fit_and_predict_candidate(
                candidate, train, train.churn, valid[list(candidate.features)]
            )
        if not np.isfinite(values).all():
            raise ValueError("Inner OOF scores do not cover every development row.")
        scores[candidate.candidate_id] = values
    return scores, assignment


def choose_candidate(data, candidates, scores, policy):
    choices = []
    for candidate in candidates:
        choice = policy.select(cutoff_scan(data.churn, scores[candidate.candidate_id]))
        if choice is not None:
            choices.append((candidate, choice))
    if not choices:
        return None

    def selection_key(item):
        candidate, choice = item
        metrics = (choice["recall"], choice["precision"])
        if policy.objective == "precision_at_recall":
            metrics = metrics[::-1]
        return (
            -metrics[0],
            -metrics[1],
            len(candidate.features),
            candidate.candidate_id,
        )

    return min(choices, key=selection_key)


def _save_inner_scores(directory, data, scores, assignments):
    parts = [
        data[[*KEYS, "churn"]].assign(
            candidate_id=candidate_id,
            score=values,
            inner_fold=assignments,
        )
        for candidate_id, values in scores.items()
    ]
    pd.concat(parts, ignore_index=True).to_csv(directory / "inner_oof.csv", index=False)


def _preflight(data, candidates, plan):
    """Fail before fitting or writing if a configured partition is infeasible."""
    outer = list(plan.outer_splits(data))
    training_parts = [data, *(data.iloc[ti] for _, _, ti, _ in outer)]
    requires_period_classes = any(
        c.imbalance_strategy == "period_class_weights" for c in candidates
    )
    for training in training_parts:
        if requires_period_classes:
            period_weights(training)
        for ti, _ in plan.inner_splits(training):
            if requires_period_classes:
                period_weights(training.iloc[ti])
    return outer


def _fit_outer_fold(data, candidates, output, plan, policies, split):
    repeat, fold, training, held = split
    parts, statuses = [], []
    train, valid = data.iloc[training], data.iloc[held]
    folder = output / f"repeat_{repeat}" / f"fold_{fold}"
    folder.mkdir(parents=True)
    scores, assignments = inner_oof_scores(train, candidates, plan)
    _save_inner_scores(folder, train, scores, assignments)
    write_json(
        folder / "split.json",
        {
            "training_indices": training.tolist(),
            "validation_indices": held.tolist(),
            "training_store_ids": sorted(train.StoreID.unique().tolist()),
            "validation_store_ids": sorted(valid.StoreID.unique().tolist()),
            "inner_seed": plan.inner_seed,
        },
    )
    fitted_scores = {}
    for index, policy in enumerate(policies):
        chosen = choose_candidate(train, candidates, scores, policy)
        status = {
            "repeat": repeat,
            "outer_fold": fold,
            "policy": index,
            "status": "selected" if chosen else "no_feasible_candidate",
        }
        if chosen is not None:
            candidate, choice = chosen
            identity = candidate.candidate_id
            if identity not in fitted_scores:
                fitted_scores[identity], _ = fit_and_predict_candidate(
                    candidate, train, train.churn, valid[list(candidate.features)]
                )
            values = fitted_scores[identity]
            status.update(candidate_id=identity, threshold=choice["cutoff"])
            part = valid[[*KEYS, "churn"]].assign(
                repeat=repeat,
                outer_fold=fold,
                policy=index,
                candidate_id=identity,
                churn_probability=values,
                threshold=choice["cutoff"],
                predicted_churn=apply_threshold(values, choice["cutoff"]),
            )
            if "n_orders_3m" in valid:
                part["n_orders_3m"] = valid.n_orders_3m.to_numpy()
            parts.append(part)
        statuses.append(status)
    print(f"Completed grouped repeat {repeat}, outer fold {fold}", flush=True)
    return parts, statuses


def _assess_outer_predictions(predictions, expected_rows):
    metrics = []
    if not predictions.empty:
        for (repeat, policy), rows in predictions.groupby(["repeat", "policy"]):
            # Cutoffs differ by outer fold; assess the retained decisions directly.
            for subgroup, group in [
                ("overall", rows),
                *(
                    (str(snapshot.date()), part)
                    for snapshot, part in rows.groupby("snapshot")
                ),
            ]:
                metrics.append(
                    {
                        "repeat": repeat,
                        "policy": policy,
                        "subgroup": subgroup,
                        "rows": len(group),
                        "complete": len(rows) == expected_rows,
                        **calculate_threshold_metrics(
                            group.churn, group.predicted_churn, 0.5
                        ),
                    }
                )
    columns = [
        "repeat",
        "policy",
        "subgroup",
        "rows",
        "complete",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "alert_count",
        "alert_rate",
    ]
    return pd.DataFrame(metrics, columns=columns)


def _save_full_development_selection(
    data, candidates, output, model_directory, plan, policies
):
    final_scores, assignments = inner_oof_scores(data, candidates, plan)
    _save_inner_scores(output, data, final_scores, assignments)
    model_directory.mkdir(parents=True)
    selection_directory = output / "selection"
    selection_directory.mkdir()
    bundles, selections = {}, []
    for index, policy in enumerate(policies):
        chosen = choose_candidate(data, candidates, final_scores, policy)
        manifest = {
            "schema": "grouped-selection-v1",
            "policy": policy.to_dict(),
            "validation_plan": plan.to_dict(),
            "data_sha256": dataframe_sha256(data),
            "labels_available_at": str(
                (data.snapshot.max() + pd.DateOffset(months=6)).date()
            ),
            "status": "selected" if chosen else "no_feasible_candidate",
            "source": source_identity(),
            "model_bundle": None,
        }
        if chosen is not None:
            candidate, choice = chosen
            identity = candidate.candidate_id
            if identity not in bundles:
                predictor = fit_candidate_ensemble(
                    candidate, data, data.churn, manifest["data_sha256"]
                )
                path = model_directory / f"{identity}.joblib"
                joblib.dump(predictor, path)
                bundles[identity] = {
                    "path": os.path.relpath(
                        path.resolve(), selection_directory.resolve()
                    ),
                    "sha256": file_sha256(path),
                }
            manifest.update(
                candidate=candidate.to_dict(),
                threshold=choice["cutoff"],
                model_bundle=bundles[identity],
                threshold_comparison=">=",
            )
        write_json(selection_directory / f"policy_{index:02d}.json", manifest)
        selections.append(manifest)
    return len(selections)


def run_grouped_study(
    data,
    candidates,
    output: Path,
    model_directory: Path,
    *,
    plan=None,
    policies=None,
    inputs=None,
):
    """Assess a selection procedure; outer outcomes never choose its configuration."""
    plan = plan or GroupedValidationPlan()
    policies = (ThresholdPolicy(),) if policies is None else tuple(policies)
    if not policies:
        raise ValueError("Provide at least one cutoff policy.")
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Provide nonempty, distinct candidate configurations.")
    if len({tuple(p.to_dict().items()) for p in policies}) != len(policies):
        raise ValueError("Cutoff policies must be distinct.")
    output, model_directory = Path(output), Path(model_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Development outputs already exist; choose a new run ID.")
    if model_directory.exists():
        raise FileExistsError("Model outputs already exist; choose a new run ID.")
    if output.resolve().is_relative_to(model_directory.resolve()):
        raise ValueError(
            "Model directory must not contain the development output directory."
        )
    data = validate_grouped_data(data, candidates)
    outer = _preflight(data, candidates, plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "protocol.json",
        {
            "validation": plan.to_dict(),
            "policies": [p.to_dict() for p in policies],
            "candidates": [c.to_dict() for c in candidates],
            "data_sha256": dataframe_sha256(data),
            "selection": "inner pooled OOF only; ties fewer features then stable candidate ID",
            "interpretation": "held-out reseller generalization within the development periods",
        },
    )
    parts, statuses = [], []
    for split in outer:
        fold_parts, fold_statuses = _fit_outer_fold(
            data, candidates, output, plan, policies, split
        )
        parts.extend(fold_parts)
        statuses.extend(fold_statuses)
    prediction_columns = [
        *KEYS,
        "churn",
        "repeat",
        "outer_fold",
        "policy",
        "candidate_id",
        "churn_probability",
        "threshold",
        "predicted_churn",
    ]
    if "n_orders_3m" in data:
        prediction_columns.append("n_orders_3m")
    predictions = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=prediction_columns)
    )
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(statuses).to_csv(output / "selection_status.csv", index=False)
    _assess_outer_predictions(predictions, len(data)).to_csv(
        output / "operating_metrics.csv", index=False
    )
    selection_count = _save_full_development_selection(
        data, candidates, output, model_directory, plan, policies
    )
    evidence = {
        str(p.relative_to(output)): file_sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_manifest(
        output,
        {
            "evidence": evidence,
            "inputs": inputs or {},
            "stage": "grouped_development",
            "validation_plan": plan.to_dict(),
            "data_sha256": dataframe_sha256(data),
            "selection_count": selection_count,
            "protected_final_test_labels_accessed": False,
        },
    )
    return output

"""Run the bounded temporal candidate comparison and precision-floor study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.project_paths import (
    MODELS_DIR,
    get_run_directory,
)
from reseller_churn.config.validation_settings import (
    DEVELOPMENT_FOLDS,
    LABELED_SNAPSHOTS,
)
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.evaluation.metrics import calculate_ranking_metrics
from reseller_churn.evaluation.reporting import (
    error_analysis_from_predictions,
    plot_precision_recall_tradeoffs,
)
from reseller_churn.evaluation.selection import (
    DEFAULT_PRECISION_FLOORS,
    SELECTION_RULE,
    evaluate_threshold_tradeoffs,
    select_across_precision_floors,
)
from reseller_churn.modeling.candidates import (
    Candidate,
    build_candidates,
)
from reseller_churn.modeling.development import (
    dataframe_sha256,
    fit_and_predict_candidate,
    fold_manifest,
    refit_selected_candidate,
    repository_identity,
    validate_labeled_development_data,
)


def _candidate_search_space(
    candidates: list[Candidate],
    tuned: bool,
) -> dict[str, object]:
    full_candidate_ids = {
        candidate.candidate_id for candidate in build_candidates(tuned=tuned)
    }
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    return {
        "tuned": tuned,
        "subset_search": set(candidate_ids) != full_candidate_ids,
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "architectures": sorted({candidate.architecture for candidate in candidates}),
        "feature_sets": sorted({candidate.feature_set for candidate in candidates}),
        "imbalance_strategies": sorted(
            {candidate.imbalance_strategy for candidate in candidates}
        ),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def _fold_metrics_for_manifest(selected: pd.Series) -> dict[str, dict[str, object]]:
    metrics_by_fold = {}
    for fold in DEVELOPMENT_FOLDS:
        metrics_by_fold[fold.name] = {
            metric_name: _json_scalar(selected[f"{fold.name}_{metric_name}"])
            for metric_name in (
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
            )
        }
    return metrics_by_fold


def _selection_manifest(
    *,
    run_id: str,
    selected: pd.Series,
    candidate: Candidate | None,
    precision_floor: float,
    data_sha256: str,
    search_space: dict[str, object],
    repository: dict[str, object],
    model_bundle: dict[str, str] | None,
    source_file_sha256: str,
) -> dict[str, object]:
    status = str(selected["status"])
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "run_id": run_id,
        "status": status,
        "minimum_precision": precision_floor,
        "threshold": (
            float(selected["threshold"]) if status == "selected" else None
        ),
        "selection_rule": SELECTION_RULE,
        "approved_training_snapshots": list(LABELED_SNAPSHOTS),
        "development_data_path": (
            f"results/{run_id}/preparation/labeled_development.csv"
        ),
        "folds": fold_manifest(),
        "data_sha256": data_sha256,
        "source_data_file_sha256": source_file_sha256,
        "search_space": search_space,
        "repository": repository,
        "precision_floor_status": "sensitivity_value_not_approved_business_minimum",
        "prediction_policy": None,
        "selected_fold_metrics": {},
        "selected_summary_metrics": {},
        "model_bundle": model_bundle,
    }
    if candidate is None:
        manifest["candidate"] = None
        manifest["message"] = (
            "No complete candidate-threshold configuration met the required "
            "precision in every fold."
        )
        return manifest

    manifest["candidate"] = candidate.to_dict()
    manifest["prediction_policy"] = {
        "seed_aggregation": "mean_predicted_probability",
        "seeds": list(candidate.seeds),
        "threshold_comparison": ">=",
    }
    manifest["selected_fold_metrics"] = _fold_metrics_for_manifest(selected)
    manifest["selected_summary_metrics"] = {
        metric_name: _json_scalar(selected.get(metric_name))
        for metric_name in (
            "mean_recall",
            "worst_fold_recall",
            "mean_precision",
            "worst_fold_precision",
            "pooled_tp",
            "pooled_fp",
            "pooled_fn",
            "pooled_tn",
            "pooled_precision",
            "pooled_recall",
            "pooled_f1",
            "pooled_alert_count",
            "pooled_alert_rate",
        )
    }
    return manifest


def _json_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_configuration_rows(candidates: list[Candidate]) -> list[dict[str, str]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "architecture": candidate.architecture,
            "feature_set": candidate.feature_set,
            "features": json.dumps(candidate.features),
            "imbalance_strategy": candidate.imbalance_strategy,
            "parameters": json.dumps(candidate.parameters, sort_keys=True),
            "seeds": json.dumps(candidate.seeds),
            "seed_aggregation": "mean_predicted_probability",
        }
        for candidate in candidates
    ]


def _collect_candidate_results(
    candidates: list[Candidate],
    temporal_data: TemporalDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every candidate once per fold and retain its validation probabilities."""
    fold_metric_rows = []
    prediction_parts = []
    tradeoff_parts = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            f"[{candidate_index}/{len(candidates)}] {candidate.candidate_id}",
            flush=True,
        )
        fold_predictions = {}
        for fold in DEVELOPMENT_FOLDS:
            prepared = temporal_data.prepare_fold(fold)
            target = prepared.validation_data["churn"].astype(int).to_numpy()
            probabilities, _ = fit_and_predict_candidate(
                candidate,
                training_features=prepared.training_data,
                training_target=prepared.training_data["churn"].astype(int),
                validation_features=prepared.validation_data,
            )
            fold_predictions[fold.name] = {
                "target": target,
                "probabilities": probabilities,
            }
            fold_metric_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "architecture": candidate.architecture,
                    "feature_set": candidate.feature_set,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "fold": fold.name,
                    **calculate_ranking_metrics(target, probabilities),
                }
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate.candidate_id,
                        "architecture": candidate.architecture,
                        "feature_set": candidate.feature_set,
                        "imbalance_strategy": candidate.imbalance_strategy,
                        "fold": fold.name,
                        "StoreID": prepared.validation_data["StoreID"].to_numpy(),
                        "snapshot": prepared.validation_data["snapshot"].to_numpy(),
                        "actual_churn": target,
                        "churn_probability": probabilities,
                    }
                )
            )

        candidate_tradeoffs = evaluate_threshold_tradeoffs(
            candidate.candidate_id,
            fold_predictions,
        )
        candidate_tradeoffs["architecture"] = candidate.architecture
        candidate_tradeoffs["feature_set"] = candidate.feature_set
        candidate_tradeoffs["imbalance_strategy"] = candidate.imbalance_strategy
        tradeoff_parts.append(candidate_tradeoffs)

    return (
        pd.DataFrame(fold_metric_rows),
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(tradeoff_parts, ignore_index=True),
    )


def _save_development_reports(
    development_directory: Path,
    candidates: list[Candidate],
    fold_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    threshold_tradeoffs: pd.DataFrame,
    floor_comparison: pd.DataFrame,
) -> None:
    development_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_candidate_configuration_rows(candidates)).to_csv(
        development_directory / "candidate_configurations.csv",
        index=False,
    )
    fold_metrics.to_csv(
        development_directory / "fold_ranking_metrics.csv",
        index=False,
    )
    validation_predictions.to_csv(
        development_directory / "validation_predictions.csv",
        index=False,
    )
    threshold_tradeoffs.to_csv(
        development_directory / "threshold_tradeoffs.csv",
        index=False,
    )
    floor_comparison.to_csv(
        development_directory / "precision_floor_comparison.csv",
        index=False,
    )
    plot_precision_recall_tradeoffs(
        threshold_tradeoffs,
        development_directory / "recall_precision_tradeoffs.png",
    )


def _refit_selected_predictors(
    *,
    run_id: str,
    labeled_data: pd.DataFrame,
    floor_comparison: pd.DataFrame,
    candidate_by_id: dict[str, Candidate],
    data_sha256: str,
    source_file_sha256: str,
    search_space: dict[str, object],
    repository: dict[str, object],
) -> dict[str, dict[str, str]]:
    selected_rows = floor_comparison[floor_comparison["status"] == "selected"]
    candidate_ids = sorted(set(selected_rows["candidate_id"].dropna()))
    if not candidate_ids:
        return {}

    models_directory = MODELS_DIR / "provisional" / run_id
    models_directory.mkdir(parents=True, exist_ok=True)
    model_bundles = {}
    for candidate_id in candidate_ids:
        selected = selected_rows[selected_rows["candidate_id"] == candidate_id].iloc[0]
        candidate = candidate_by_id[candidate_id]
        selected_manifest = _selection_manifest(
            run_id=run_id,
            selected=selected,
            candidate=candidate,
            precision_floor=float(selected["minimum_precision"]),
            data_sha256=data_sha256,
            source_file_sha256=source_file_sha256,
            search_space=search_space,
            repository=repository,
            model_bundle=None,
        )
        predictor = refit_selected_candidate(
            labeled_data,
            selected_manifest,
            data_sha256,
        )
        model_path = models_directory / f"{candidate_id}.joblib"
        joblib.dump(predictor, model_path)
        model_bundles[candidate_id] = {
            "path": model_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(model_path),
        }
    return model_bundles


def _write_selection_outputs(
    *,
    run_id: str,
    development_directory: Path,
    floor_comparison: pd.DataFrame,
    candidate_by_id: dict[str, Candidate],
    model_bundles: dict[str, dict[str, str]],
    validation_predictions: pd.DataFrame,
    data_sha256: str,
    source_file_sha256: str,
    search_space: dict[str, object],
    repository: dict[str, object],
) -> None:
    selection_directory = development_directory / "selection"
    selection_directory.mkdir()
    for floor_index, (_, selected) in enumerate(floor_comparison.iterrows(), start=1):
        precision_floor = float(selected["minimum_precision"])
        candidate_id = selected.get("candidate_id")
        candidate = (
            candidate_by_id.get(candidate_id)
            if isinstance(candidate_id, str)
            else None
        )
        manifest = _selection_manifest(
            run_id=run_id,
            selected=selected,
            candidate=candidate,
            precision_floor=precision_floor,
            data_sha256=data_sha256,
            search_space=search_space,
            repository=repository,
            model_bundle=model_bundles.get(candidate_id),
            source_file_sha256=source_file_sha256,
        )
        floor_label = f"{precision_floor:.12g}"
        write_json(
            selection_directory / f"precision_{floor_index:02d}_{floor_label}.json",
            manifest,
        )

        if candidate is not None:
            error_report = error_analysis_from_predictions(
                validation_predictions,
                candidate_id,
                float(selected["threshold"]),
            )
            error_report.to_csv(
                development_directory
                / f"error_analysis_precision_{floor_index:02d}_{floor_label}.csv",
                index=False,
            )


def run_development(
    run_id: str,
    *,
    precision_floors: tuple[float, ...] = DEFAULT_PRECISION_FLOORS,
    tuned: bool = False,
    architectures: tuple[str, ...] | None = None,
    feature_sets: tuple[str, ...] | None = None,
    imbalance_strategies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Compare candidates, report each requested floor, and refit unique winners."""
    run_directory = get_run_directory(run_id)
    preparation_directory = run_directory / "preparation"
    development_directory = run_directory / "development"
    labeled_data_path = preparation_directory / "labeled_development.csv"
    if not labeled_data_path.is_file():
        raise FileNotFoundError(
            f"Prepared labeled development data not found: {labeled_data_path}"
        )
    if not development_directory.is_dir():
        raise FileNotFoundError(
            f"Run preparation first; missing {development_directory}."
        )
    if any(development_directory.iterdir()):
        raise FileExistsError(
            "Development outputs already exist for this run. "
            "Choose a new run ID to preserve previous results."
        )
    if not precision_floors:
        raise ValueError("At least one precision floor is required.")
    if any(not np.isfinite(floor) or not 0 <= floor <= 1 for floor in precision_floors):
        raise ValueError("Precision floors must be finite values between 0 and 1.")
    if len(set(precision_floors)) != len(precision_floors):
        raise ValueError("Precision floors must be unique.")

    labeled_data = validate_labeled_development_data(
        DatasetLoader.load_labeled_snapshots(labeled_data_path)
    )
    data_sha256 = dataframe_sha256(labeled_data)
    source_file_sha256 = file_sha256(labeled_data_path)
    candidates = build_candidates(
        tuned=tuned,
        architectures=architectures,
        feature_sets=feature_sets,
        imbalance_strategies=imbalance_strategies,
        seeds=seeds,
    )
    if not candidates:
        raise ValueError("The candidate search space is empty.")
    search_space = _candidate_search_space(candidates, tuned)
    repository = repository_identity()

    print(f"Run ID: {run_id}")
    print(f"Candidate configurations: {len(candidates)}")
    print(f"Precision-floor sensitivity values: {list(precision_floors)}")
    print("Protected final-test labels: NOT ACCESSED")
    fold_metrics, validation_predictions, threshold_tradeoffs = (
        _collect_candidate_results(candidates, TemporalDataset(labeled_data))
    )
    floor_comparison = select_across_precision_floors(
        threshold_tradeoffs,
        precision_floors,
    )
    _save_development_reports(
        development_directory,
        candidates,
        fold_metrics,
        validation_predictions,
        threshold_tradeoffs,
        floor_comparison,
    )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    model_bundles = _refit_selected_predictors(
        run_id=run_id,
        labeled_data=labeled_data,
        floor_comparison=floor_comparison,
        candidate_by_id=candidate_by_id,
        data_sha256=data_sha256,
        source_file_sha256=source_file_sha256,
        search_space=search_space,
        repository=repository,
    )
    _write_selection_outputs(
        run_id=run_id,
        development_directory=development_directory,
        floor_comparison=floor_comparison,
        candidate_by_id=candidate_by_id,
        model_bundles=model_bundles,
        validation_predictions=validation_predictions,
        data_sha256=data_sha256,
        source_file_sha256=source_file_sha256,
        search_space=search_space,
        repository=repository,
    )

    report_paths = {
        "candidate_configurations": "candidate_configurations.csv",
        "fold_ranking_metrics": "fold_ranking_metrics.csv",
        "validation_predictions": "validation_predictions.csv",
        "threshold_tradeoffs": "threshold_tradeoffs.csv",
        "precision_floor_comparison": "precision_floor_comparison.csv",
    }

    run_manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "purpose": "precision-floor sensitivity experiment",
        "precision_floors": list(precision_floors),
        "search_space": search_space,
        "data_sha256": data_sha256,
        "source_data_file_sha256": source_file_sha256,
        "reports": report_paths,
        "selection_status_by_floor": [
            {
                "minimum_precision": float(row["minimum_precision"]),
                "status": str(row["status"]),
            }
            for _, row in floor_comparison.iterrows()
        ],
        "protected_final_test_labels_accessed": False,
        "repository": repository,
    }
    write_json(development_directory / "development_manifest.json", run_manifest)

    print()
    display_columns = [
        "minimum_precision",
        "status",
        "candidate_id",
        "architecture",
        "feature_set",
        "imbalance_strategy",
        "threshold",
        "mean_recall",
        "worst_fold_recall",
        "mean_precision",
        "worst_fold_precision",
        "pooled_alert_count",
    ]
    display_columns = [
        column for column in display_columns if column in floor_comparison.columns
    ]
    print(floor_comparison[display_columns].to_string(index=False))
    print(f"Saved development reports: {development_directory}")
    print("Precision floors are sensitivity scenarios; no floor is an approved business minimum.")
    if model_bundles:
        print(f"Saved unique refitted predictors: {MODELS_DIR / 'provisional' / run_id}")
    else:
        print("No precision floor selected a feasible candidate; no model was refit.")
    return development_directory

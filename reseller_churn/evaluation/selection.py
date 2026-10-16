"""Precision eligibility and deterministic recall-first selection helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from reseller_churn.evaluation.metrics import calculate_threshold_metrics


SELECTION_RULE = (
    "maximize mean fold recall subject to the minimum precision being met "
    "in every development fold; break ties by worst-fold recall, mean "
    "precision, then stable candidate-threshold ID"
)
DEFAULT_PRECISION_FLOORS = tuple(step / 10 for step in range(1, 10))
THRESHOLDS = tuple(step / 100 for step in range(101))


def _threshold_label(threshold: float) -> str:
    """Keep familiar labels while distinguishing thresholds beyond 2 decimals."""
    threshold_value = float(threshold)
    two_decimal_label = f"{threshold_value:.2f}"
    if float(two_decimal_label) == threshold_value:
        return two_decimal_label
    return repr(threshold_value)


def evaluate_threshold_tradeoffs(
    candidate_id: str,
    fold_predictions: Mapping[str, Mapping[str, object]],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """Calculate per-fold and pooled metrics once for every threshold."""
    if not fold_predictions:
        raise ValueError("At least one fold prediction is required.")

    prepared = {}
    for fold_name, values in fold_predictions.items():
        raw_target = np.asarray(values["target"])
        if raw_target.ndim != 1 or not np.isin(raw_target, [0, 1]).all():
            raise ValueError(f"{fold_name}: target must be a one-dimensional 0/1 array.")
        target = raw_target.astype(int)
        probabilities = np.asarray(values["probabilities"], dtype=float)
        if (
            probabilities.ndim != 1
            or len(target) == 0
            or len(target) != len(probabilities)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError(
                f"{fold_name}: probabilities must be finite 1D values in [0, 1] "
                "and match the non-empty target."
            )
        prepared[fold_name] = (target, probabilities)

    rows = []
    for threshold in thresholds:
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "threshold": float(threshold),
            "selection_id": (
                f"{candidate_id}__threshold-{_threshold_label(threshold)}"
            ),
        }
        fold_metrics = []
        all_targets = []
        all_probabilities = []

        for fold_name, (target, probabilities) in prepared.items():
            metrics = calculate_threshold_metrics(
                target,
                probabilities,
                threshold,
            )
            fold_metrics.append(metrics)
            all_targets.append(target)
            all_probabilities.append(probabilities)
            for metric_name, value in metrics.items():
                row[f"{fold_name}_{metric_name}"] = value

        pooled_metrics = calculate_threshold_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_probabilities),
            threshold,
        )
        for metric_name, value in pooled_metrics.items():
            row[f"pooled_{metric_name}"] = value

        recalls = [float(metrics["recall"]) for metrics in fold_metrics]
        precisions = [float(metrics["precision"]) for metrics in fold_metrics]
        row["mean_recall"] = float(np.mean(recalls))
        row["worst_fold_recall"] = float(np.min(recalls))
        row["mean_precision"] = float(np.mean(precisions))
        row["worst_fold_precision"] = float(np.min(precisions))
        row["mean_alert_count"] = float(
            np.mean([int(metrics["alert_count"]) for metrics in fold_metrics])
        )
        rows.append(row)

    return pd.DataFrame(rows)


def select_for_precision_floor(
    threshold_results: pd.DataFrame,
    minimum_precision: float,
) -> dict[str, object]:
    """Select the best candidate-threshold row for one experimental floor."""
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be between 0 and 1.")
    if threshold_results.empty:
        return {
            "status": "no_feasible_candidate",
            "minimum_precision": minimum_precision,
            "selection_rule": SELECTION_RULE,
        }

    fold_precision_columns = [
        column
        for column in threshold_results.columns
        if column.endswith("_precision")
        and not column.startswith(("pooled_", "mean_", "worst_"))
    ]
    if not fold_precision_columns:
        raise ValueError("Threshold results do not contain per-fold precision.")

    eligible = threshold_results[
        threshold_results[fold_precision_columns]
        .ge(minimum_precision)
        .all(axis=1)
    ]
    if eligible.empty:
        return {
            "status": "no_feasible_candidate",
            "minimum_precision": minimum_precision,
            "selection_rule": SELECTION_RULE,
            "candidate_id": None,
            "threshold": None,
        }

    selected = eligible.sort_values(
        [
            "mean_recall",
            "worst_fold_recall",
            "mean_precision",
            "selection_id",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    result = selected.to_dict()
    result.update(
        {
            "status": "selected",
            "minimum_precision": minimum_precision,
            "selection_rule": SELECTION_RULE,
        }
    )
    return result


def select_across_precision_floors(
    threshold_results: pd.DataFrame,
    precision_floors: tuple[float, ...],
) -> pd.DataFrame:
    """Build one comparable result row per requested precision floor."""
    if not precision_floors:
        raise ValueError("At least one precision floor is required.")
    if len(set(precision_floors)) != len(precision_floors):
        raise ValueError("Precision floors must be unique.")

    rows = []
    for floor in precision_floors:
        selected = select_for_precision_floor(threshold_results, floor)
        row: dict[str, object] = dict(selected)
        row.setdefault("candidate_id", None)
        row.setdefault("threshold", None)
        rows.append(row)

    return pd.DataFrame(rows)

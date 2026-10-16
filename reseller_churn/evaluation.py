"""Descriptive cutoff reports from saved scores and supplied complete outcomes."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    write_manifest,
)
from reseller_churn.config import ORIGINAL_RULES, ORIGINAL_TEST_SNAPSHOT
from reseller_churn.data import KEYS, DateLike, date, require_columns, unique_ids


def _validated_cutoffs(cutoffs: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in cutoffs)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(value) or not 0 <= value <= 1 for value in values)
    ):
        raise ValueError("Cutoffs must be unique finite values in [0, 1].")
    return values


def _validated_cases(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcomes_through: DateLike,
    cohort_status: str,
) -> pd.DataFrame:
    """Validate evaluation inputs and return the matched account population."""
    if cohort_status not in {"inspected", "independent"}:
        raise ValueError("Declare whether the cohort was previously inspected.")
    predictions, outcomes = predictions.copy(), outcomes.copy()
    require_columns(
        predictions, [*KEYS, "churn_probability", "n_orders_3m"], "predictions"
    )
    require_columns(outcomes, [*KEYS, "actual_churn"], "outcomes")
    if predictions.empty or outcomes.empty:
        raise ValueError("Evaluation populations cannot be empty.")
    for frame, name in [(predictions, "predictions"), (outcomes, "outcomes")]:
        frame["snapshot"] = frame.snapshot.map(date)
        unique_ids(frame, KEYS, name)
    if "actual_churn" in predictions or "churn" in predictions:
        raise ValueError("Pass unlabeled predictions separately from outcomes.")
    if not outcomes.actual_churn.isin([0, 1]).all():
        raise ValueError("Outcomes must contain nonmissing binary labels.")
    scores = predictions.churn_probability.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be finite in [0, 1].")
    counts = predictions.n_orders_3m.to_numpy(dtype=float)
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Recent order counts must be nonnegative integers.")
    if (
        cohort_status == "independent"
        and predictions.snapshot.eq(date(ORIGINAL_TEST_SNAPSHOT)).any()
    ):
        raise ValueError(
            "October 2013 outcomes are already inspected, not independent."
        )
    outcomes_through = date(outcomes_through)
    if any(
        snapshot + pd.DateOffset(months=6) > outcomes_through
        for snapshot in predictions.snapshot.unique()
    ):
        raise ValueError("Incomplete six-month outcome coverage.")
    cases = predictions.merge(
        outcomes[[*KEYS, "actual_churn"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not cases["_merge"].eq("both").all():
        raise ValueError("Prediction and outcome populations differ.")
    return cases.drop(columns="_merge")


def _cutoff_decisions(cohort: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Classify each account at one cutoff for the detailed decision output."""
    decision = cohort.copy()
    decision["cutoff"] = cutoff
    decision["predicted_churn"] = decision.churn_probability.ge(cutoff).astype(int)
    positive = decision.actual_churn.eq(1)
    alert = decision.predicted_churn.eq(1)
    decision["result_type"] = np.select(
        [positive & alert, positive & ~alert, ~positive & alert],
        ["caught_churn", "missed_churn", "false_alert"],
        default="correct_non_churn",
    )
    return decision


def _cohort_subgroups(cohort: pd.DataFrame) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("overall", cohort),
        ("purchased_in_3m", cohort[cohort.n_orders_3m.gt(0)]),
        ("no_purchase_in_3m", cohort[cohort.n_orders_3m.eq(0)]),
    )


def _metric_row(
    snapshot: pd.Timestamp,
    cutoff: float,
    cohort_status: str,
    subgroup: str,
    group: pd.DataFrame,
) -> dict[str, object]:
    positive = group.actual_churn.eq(1)
    alert = group.churn_probability.ge(cutoff)
    tp = int((positive & alert).sum())
    fp = int((~positive & alert).sum())
    fn = int((positive & ~alert).sum())
    tn = int((~positive & ~alert).sum())
    original_role = (
        ORIGINAL_RULES.get(cutoff, "additional")
        if snapshot == date(ORIGINAL_TEST_SNAPSHOT)
        else "user_supplied"
    )
    return {
        "snapshot": snapshot,
        "cutoff": cutoff,
        "original_evaluation_role": original_role,
        "cohort_status": cohort_status,
        "subgroup": subgroup,
        "rows": len(group),
        "churners": int(positive.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "alert_count": tp + fp,
        "alert_rate": (tp + fp) / len(group),
        "average_precision": float(
            average_precision_score(positive, group.churn_probability)
        )
        if positive.any()
        else None,
        "roc_auc": float(roc_auc_score(positive, group.churn_probability))
        if positive.nunique() == 2
        else None,
    }


def compare_cutoffs(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    cutoffs: Iterable[float],
    outcomes_through: DateLike,
    cohort_status: str = "inspected",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cutoffs on matched populations with mature observed outcomes."""
    cutoffs = _validated_cutoffs(cutoffs)
    cases = _validated_cases(predictions, outcomes, outcomes_through, cohort_status)
    rows, decisions = [], []
    for snapshot, cohort in cases.groupby("snapshot", sort=True):
        for cutoff in cutoffs:
            decisions.append(_cutoff_decisions(cohort, cutoff))
            for subgroup, group in _cohort_subgroups(cohort):
                if group.empty:
                    continue
                rows.append(
                    _metric_row(snapshot, cutoff, cohort_status, subgroup, group)
                )
    return pd.DataFrame(rows), pd.concat(decisions, ignore_index=True)


def evaluate(
    predictions_path: Path,
    outcomes_path: Path,
    output: Path,
    cutoffs,
    outcomes_through,
    cohort_status,
    approve_independent=False,
) -> Path:
    if cohort_status == "independent" and not approve_independent:
        raise ValueError(
            "Independent evaluation requires explicit approval and predeclared methodology/cutoffs."
        )
    metrics, decisions = compare_cutoffs(
        pd.read_csv(predictions_path, float_precision="round_trip"),
        pd.read_csv(outcomes_path),
        cutoffs,
        outcomes_through,
        cohort_status,
    )
    output = new_output_directory(output)
    metrics.to_csv(output / "cutoff_metrics.csv", index=False)
    decisions.to_csv(output / "cutoff_decisions.csv", index=False)
    write_manifest(
        output,
        {
            "stage": "evaluate",
            "inputs": input_identity(
                {"predictions": predictions_path, "outcomes": outcomes_path}
            ),
            "cutoffs": list(cutoffs),
            "outcomes_through": str(date(outcomes_through).date()),
            "cohort_status": cohort_status,
            "models_fitted": 0,
            "thresholds_optimized": False,
            "independent_evaluation_approved": bool(approve_independent),
            "interpretation": "descriptive reuse of inspected outcomes"
            if cohort_status == "inspected"
            else "user-declared independent cohort; freeze and chronology must be established externally",
        },
    )
    return output

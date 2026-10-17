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
from reseller_churn.config import KEYS, ORIGINAL_RULES, ORIGINAL_TEST_SNAPSHOT
from reseller_churn.data import (
    require_columns,
    require_unique_keys,
)
from reseller_churn.dates import DateLike, parse_calendar_date


def _validate_cutoffs(cutoffs: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in cutoffs)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(value) or not 0 <= value <= 1 for value in values)
    ):
        raise ValueError("Cutoffs must be unique finite values in [0, 1].")
    return values


def _validate_evaluation_inputs(
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
        frame["snapshot"] = frame.snapshot.map(parse_calendar_date)
        require_unique_keys(frame, KEYS, name)
    if "actual_churn" in predictions or "churn" in predictions:
        raise ValueError("Pass unlabeled predictions separately from outcomes.")
    if not outcomes.actual_churn.isin([0, 1]).all():
        raise ValueError("Outcomes must contain nonmissing binary labels.")
    _validate_prediction_values(predictions)
    if (
        cohort_status == "independent"
        and predictions.snapshot.eq(parse_calendar_date(ORIGINAL_TEST_SNAPSHOT)).any()
    ):
        raise ValueError(
            "October 2013 outcomes are already inspected, not independent."
        )
    outcomes_through = parse_calendar_date(outcomes_through)
    if any(
        snapshot + pd.DateOffset(months=6) > outcomes_through
        for snapshot in predictions.snapshot.unique()
    ):
        raise ValueError("Incomplete six-month outcome coverage.")
    return _match_predictions_and_outcomes(predictions, outcomes)


def _validate_prediction_values(predictions: pd.DataFrame) -> None:
    """Require finite probabilities and nonnegative integer recent-order counts."""
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


def _match_predictions_and_outcomes(
    predictions: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Require exactly one outcome per scored account, with no extra outcomes."""
    matched_accounts = predictions.merge(
        outcomes[[*KEYS, "actual_churn"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not matched_accounts["_merge"].eq("both").all():
        raise ValueError("Prediction and outcome populations differ.")
    return matched_accounts.drop(columns="_merge")


def _cutoff_decisions(cohort: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Classify each account at one cutoff for the detailed decisions output."""
    decisions = cohort.copy()
    decisions["cutoff"] = cutoff
    decisions["predicted_churn"] = decisions.churn_probability.ge(cutoff).astype(int)
    actually_churned = decisions.actual_churn.eq(1)
    was_alerted = decisions.predicted_churn.eq(1)
    decisions["result_type"] = np.select(
        [
            actually_churned & was_alerted,
            actually_churned & ~was_alerted,
            ~actually_churned & was_alerted,
        ],
        ["caught_churn", "missed_churn", "false_alert"],
        default="correct_non_churn",
    )
    return decisions


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
    accounts: pd.DataFrame,
) -> dict[str, object]:
    """Summarize one nonempty snapshot/subgroup at one supplied cutoff."""
    actually_churned = accounts.actual_churn.eq(1)
    was_alerted = accounts.churn_probability.ge(cutoff)
    true_positives = int((actually_churned & was_alerted).sum())
    false_positives = int((~actually_churned & was_alerted).sum())
    false_negatives = int((actually_churned & ~was_alerted).sum())
    true_negatives = int((~actually_churned & ~was_alerted).sum())
    alert_count = true_positives + false_positives
    churn_count = true_positives + false_negatives

    # Preserve the reporting convention: zero when a threshold metric is undefined.
    precision = true_positives / max(1, alert_count)
    recall = true_positives / max(1, churn_count)
    f1 = (
        2
        * true_positives
        / max(1, 2 * true_positives + false_positives + false_negatives)
    )
    average_precision = None
    if actually_churned.any():
        average_precision = float(
            average_precision_score(actually_churned, accounts.churn_probability)
        )
    roc_auc = None
    if actually_churned.nunique() == 2:
        roc_auc = float(roc_auc_score(actually_churned, accounts.churn_probability))

    if snapshot == parse_calendar_date(ORIGINAL_TEST_SNAPSHOT):
        original_role = ORIGINAL_RULES.get(cutoff, "additional")
    else:
        original_role = "user_supplied"

    return {
        "snapshot": snapshot,
        "cutoff": cutoff,
        "original_evaluation_role": original_role,
        "cohort_status": cohort_status,
        "subgroup": subgroup,
        "rows": len(accounts),
        "churners": int(actually_churned.sum()),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tn": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_count": alert_count,
        "alert_rate": alert_count / len(accounts),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def compare_cutoffs(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    cutoffs: Iterable[float],
    outcomes_through: DateLike,
    cohort_status: str = "inspected",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cutoffs on matched populations with mature observed outcomes."""
    validated_cutoffs = _validate_cutoffs(cutoffs)
    matched_accounts = _validate_evaluation_inputs(
        predictions, outcomes, outcomes_through, cohort_status
    )
    metric_rows = []
    decisions_by_snapshot_and_cutoff = []
    for snapshot, cohort in matched_accounts.groupby("snapshot", sort=True):
        for cutoff in validated_cutoffs:
            decisions_by_snapshot_and_cutoff.append(_cutoff_decisions(cohort, cutoff))
            for subgroup_name, subgroup_accounts in _cohort_subgroups(cohort):
                if subgroup_accounts.empty:
                    continue
                metric_rows.append(
                    _metric_row(
                        snapshot,
                        cutoff,
                        cohort_status,
                        subgroup_name,
                        subgroup_accounts,
                    )
                )
    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decisions_by_snapshot_and_cutoff, ignore_index=True)
    return metrics, decisions


def evaluate(
    predictions_path: Path,
    outcomes_path: Path,
    output: Path,
    cutoffs: Iterable[float],
    outcomes_through: DateLike,
    cohort_status: str,
    approve_independent: bool = False,
) -> Path:
    """Save descriptive cutoff reports from separate prediction and outcome CSVs."""
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
    output_directory = new_output_directory(output)
    metrics.to_csv(output_directory / "cutoff_metrics.csv", index=False)
    decisions.to_csv(output_directory / "cutoff_decisions.csv", index=False)
    if cohort_status == "inspected":
        interpretation = "descriptive reuse of inspected outcomes"
    else:
        interpretation = "user-declared independent cohort; freeze and chronology must be established externally"
    write_manifest(
        output_directory,
        {
            "stage": "evaluate",
            "inputs": input_identity(
                {"predictions": predictions_path, "outcomes": outcomes_path}
            ),
            "cutoffs": list(cutoffs),
            "outcomes_through": str(parse_calendar_date(outcomes_through).date()),
            "cohort_status": cohort_status,
            "models_fitted": 0,
            "thresholds_optimized": False,
            "independent_evaluation_approved": bool(approve_independent),
            "interpretation": interpretation,
        },
    )
    return output_directory

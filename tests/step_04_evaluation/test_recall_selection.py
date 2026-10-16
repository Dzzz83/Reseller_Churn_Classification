import pandas as pd

from reseller_churn.evaluation.selection import (
    evaluate_threshold_tradeoffs,
    select_across_precision_floors,
    select_for_precision_floor,
)


def test_precision_floor_requires_every_fold_and_uses_recall_tie_breaks():
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "a_low_mean_precision",
                "selection_id": "a_low_mean_precision__0.40",
                "threshold": 0.4,
                "fold_1_precision": 0.75,
                "fold_2_precision": 0.45,
                "mean_recall": 0.95,
                "worst_fold_recall": 0.9,
                "mean_precision": 0.60,
            },
            {
                "candidate_id": "b_lower_recall",
                "selection_id": "b_lower_recall__0.40",
                "threshold": 0.4,
                "fold_1_precision": 0.65,
                "fold_2_precision": 0.65,
                "mean_recall": 0.7,
                "worst_fold_recall": 0.6,
                "mean_precision": 0.65,
            },
            {
                "candidate_id": "c_best_recall",
                "selection_id": "c_best_recall__0.50",
                "threshold": 0.5,
                "fold_1_precision": 0.7,
                "fold_2_precision": 0.7,
                "mean_recall": 0.8,
                "worst_fold_recall": 0.7,
                "mean_precision": 0.7,
            },
        ]
    )

    selected = select_for_precision_floor(rows, minimum_precision=0.6)

    assert selected["status"] == "selected"
    assert selected["candidate_id"] == "c_best_recall"
    assert selected["threshold"] == 0.5


def test_flag_everyone_fails_a_precision_floor_above_churn_prevalence():
    target = [1, 0, 0, 0]
    probabilities = [0.5, 0.5, 0.5, 0.5]
    tradeoffs = evaluate_threshold_tradeoffs(
        "flag_everyone",
        {
            "fold_1": {"target": target, "probabilities": probabilities},
            "fold_2": {"target": target, "probabilities": probabilities},
        },
    )

    comparison = select_across_precision_floors(
        tradeoffs,
        precision_floors=(0.2, 0.3),
    )

    assert comparison.loc[0, "status"] == "selected"
    assert comparison.loc[0, "mean_precision"] == 0.25
    assert comparison.loc[1, "status"] == "no_feasible_candidate"
    assert pd.isna(comparison.loc[1, "candidate_id"])


def test_selection_is_stable_when_recall_precision_and_worst_fold_tie():
    rows = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "selection_id": f"{candidate_id}__threshold-0.50",
                "threshold": 0.5,
                "fold_1_precision": 0.7,
                "fold_2_precision": 0.7,
                "mean_recall": 0.8,
                "worst_fold_recall": 0.6,
                "mean_precision": 0.7,
            }
            for candidate_id in ("candidate_b", "candidate_a")
        ]
    )

    selected = select_for_precision_floor(rows, minimum_precision=0.6)

    assert selected["candidate_id"] == "candidate_a"


def test_selection_is_order_invariant_for_distinct_sub_cent_thresholds():
    folds = {
        "fold_1": {"target": [1, 0], "probabilities": [0.8, 0.2]},
        "fold_2": {"target": [1, 0], "probabilities": [0.8, 0.2]},
    }
    thresholds = (0.401, 0.404)
    forward = evaluate_threshold_tradeoffs(
        "same_predictions",
        folds,
        thresholds=thresholds,
    )
    reverse = evaluate_threshold_tradeoffs(
        "same_predictions",
        folds,
        thresholds=tuple(reversed(thresholds)),
    )

    selected_forward = select_for_precision_floor(forward, minimum_precision=0.5)
    selected_reverse = select_for_precision_floor(reverse, minimum_precision=0.5)

    assert forward["selection_id"].nunique() == 2
    assert selected_forward["threshold"] == selected_reverse["threshold"] == 0.401

import numpy as np
import pandas as pd
import pytest

from comparisons.validation_strategy_experiment import (
    Candidate,
    EXPERIMENT_FEATURE_SETS,
    _selection_folds,
    build_labeled_snapshots,
    candidate_manifest,
    determine_winner,
    metrics_at_budget,
)
from reseller_churn.data.dataset_loader import DatasetLoader


def test_candidate_manifest_is_complete_and_unique():
    candidates = candidate_manifest()

    assert len(candidates) == 36
    assert len({candidate.name for candidate in candidates}) == 36


def test_unverified_store_attribute_is_excluded_from_all_candidates():
    assert all("store_age" not in features for features in EXPERIMENT_FEATURE_SETS.values())


def test_selection_rejects_common_evaluation_rows():
    frame = pd.DataFrame({
        "snapshot": pd.to_datetime(["2012-04-01", "2012-10-01", "2013-04-01"]),
        "StoreID": [1, 2, 3],
    })
    for strategy in ("temporal", "group_kfold"):
        with pytest.raises(AssertionError, match="unexpected snapshots"):
            _selection_folds(frame, strategy)


def test_group_folds_keep_repeated_resellers_together():
    frame = pd.DataFrame({
        "StoreID": list(range(10)) * 2,
        "snapshot": pd.to_datetime(["2012-04-01"] * 10 + ["2012-10-01"] * 10),
    })
    folds = _selection_folds(frame, "group_kfold")
    validation_indices = []
    for _, training, validation in folds:
        assert not set(frame.iloc[training]["StoreID"]) & set(frame.iloc[validation]["StoreID"])
        validation_indices.extend(validation.tolist())
    assert sorted(validation_indices) == list(range(len(frame)))


def test_temporal_fold_uses_only_resolved_april_training_labels():
    frame = pd.DataFrame({
        "StoreID": [1, 2, 1, 2],
        "snapshot": pd.to_datetime(["2012-04-01"] * 2 + ["2012-10-01"] * 2),
    })
    _, training, validation = _selection_folds(frame, "temporal")[0]
    assert training.tolist() == [0, 1]
    assert validation.tolist() == [2, 3]


def test_protected_snapshot_is_rejected_before_feature_building():
    with pytest.raises(ValueError, match="Protected"):
        build_labeled_snapshots(
            DatasetLoader.load_orders(),
            DatasetLoader.load_stores(),
            ("2013-10-01",),
        )


def test_budget_metrics_use_exact_ceiling_alert_count_and_stable_ties():
    metrics = metrics_at_budget(
        target=np.array([1, 0, 1, 0, 0]),
        probability=np.array([0.9, 0.8, 0.8, 0.1, 0.0]),
        budget=0.4,
        tie_breaker=np.array([5, 1, 2, 3, 4]),
    )

    assert metrics == {
        "budget": 0.4,
        "alert_count": 2,
        "true_positives": 1,
        "false_positives": 1,
        "recall": 0.5,
        "precision": 0.5,
        "f1": 0.5,
    }


def test_candidate_complexity_rank_prefers_simpler_exact_tie():
    simple = Candidate("logistic", "rfm", "none")
    complex_candidate = Candidate(
        "gradient_boosting",
        "full",
        "random_oversampling",
    )

    assert simple.complexity_rank < complex_candidate.complexity_rank


def test_winner_rule_requires_budget_dominance_and_interval_support():
    rows = []
    for budget, group_recall, temporal_recall in [
        (0.1, 0.30, 0.20),
        (0.2, 0.50, 0.40),
        (0.3, 0.70, 0.70),
    ]:
        rows.extend(
            [
                {"budget": budget, "strategy": "group_kfold", "recall": group_recall},
                {"budget": budget, "strategy": "temporal", "recall": temporal_recall},
            ]
        )

    evaluation = pd.DataFrame(rows)

    assert determine_winner(evaluation, (0.01, 0.20)) == "group_kfold"
    assert determine_winner(evaluation, (-0.01, 0.20)) == "inconclusive"

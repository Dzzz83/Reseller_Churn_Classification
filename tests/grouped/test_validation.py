import numpy as np
import pytest

from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
    cutoff_scan,
)


def test_historical_search_requires_only_its_original_feature_catalog():
    from reseller_churn.features.feature_pipeline import OUTPUT_COLUMNS
    from reseller_churn.modeling.candidates import build_candidates

    historical = build_candidates()
    assert len(historical) == 36
    assert all(
        set(candidate.features).issubset(OUTPUT_COLUMNS) for candidate in historical
    )
    supported = build_candidates(
        architectures=("hist_gradient_boosting",),
        feature_sets=("ten",),
        imbalance_strategies=("period_class_weights",),
    )
    assert len(supported) == 1 and "weighted_discount_3m" in supported[0].features


def test_repeated_groups_are_reproducible_and_disjoint_at_both_levels(grouped_data):
    plan = GroupedValidationPlan(outer_folds=3, inner_folds=2, repeat_seeds=(42, 78))
    first = list(plan.outer_splits(grouped_data))
    second = list(plan.outer_splits(grouped_data))
    for (repeat, fold, ti, vi), (_, _, other_ti, other_vi) in zip(first, second):
        np.testing.assert_array_equal(ti, other_ti)
        np.testing.assert_array_equal(vi, other_vi)
        train = grouped_data.iloc[ti]
        assert set(train.StoreID).isdisjoint(grouped_data.iloc[vi].StoreID)
        for inner_ti, inner_vi in plan.inner_splits(train):
            assert set(train.iloc[inner_ti].StoreID).isdisjoint(
                train.iloc[inner_vi].StoreID
            )
    for repeat in plan.repeat_seeds:
        rows = np.concatenate([vi for seed, _, _, vi in first if seed == repeat])
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(grouped_data)))


def test_cutoff_objectives_preserve_their_distinct_selection_rules():
    target = [1] * 10 + [0, 0]
    scores = [0.1 * i for i in range(1, 11)] + [0.2, 0.05]
    scan = cutoff_scan(target, scores)
    chosen = ThresholdPolicy(objective="precision_at_recall").select(scan)
    assert chosen["cutoff"] == pytest.approx(0.1)
    assert chosen["recall"] == 1
    high_precision = ThresholdPolicy(minimum_precision=1.0).select(scan)
    assert high_precision["precision"] == 1
    assert high_precision["recall"] == 0.8
    assert scan.loc[scan.cutoff.eq(0.2), "tp"].item() == 9


def test_infeasible_cutoff_remains_explicit():
    assert (
        ThresholdPolicy(minimum_precision=1).select(cutoff_scan([1, 0], [0.2, 0.8]))
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(outer_folds=1),
        dict(inner_folds=1),
        dict(repeat_seeds=()),
        dict(repeat_seeds=(42, 42)),
        dict(inner_seed=-1),
    ],
)
def test_invalid_group_plans_rejected(kwargs):
    with pytest.raises(ValueError):
        GroupedValidationPlan(**kwargs)

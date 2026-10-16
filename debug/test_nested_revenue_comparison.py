"""Nested identity isolation, cutoff selection, and paired uncertainty checks."""

import numpy as np
import pandas as pd
import pytest

import nested_revenue_comparison as study
from test_compare_seven_ten_features import synthetic_data


def test_outer_identities_never_enter_inner_training_or_selection():
    data = synthetic_data().sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    held_out = []
    for train, validation, inner in study.nested_splits(data, 42):
        excluded = set(data.iloc[validation].StoreID)
        inner_held_out = []
        for left, right in inner:
            assert set(data.iloc[left].StoreID).isdisjoint(data.iloc[right].StoreID)
            assert excluded.isdisjoint(data.iloc[np.concatenate([left, right])].StoreID)
            inner_held_out.extend(right)
        assert sorted(inner_held_out) == sorted(train)
        held_out.extend(validation)
    assert sorted(held_out) == list(range(len(data)))


def test_inner_cutoff_requires_floor_in_every_fold():
    rows = []
    for fold in range(5):
        for label, score in [(1, .8), (0, .4), (0, .1), (0, .1)]:
            rows.append({"inner_fold": fold, "actual_churn": label, "churn_probability": score})
    selection, _ = study.select_inner(pd.DataFrame(rows), "example")
    assert selection["status"] == "selected"
    # Equal recall is resolved by higher precision, then the first stable cutoff.
    assert selection["threshold"] == .41
    assert all(selection[f"{fold}_precision"] >= .3 for fold in range(5))


def example_predictions():
    rows = []
    for repeat in [42, 78, 88]:
        for definition in ["subtotal", "totaldue"]:
            for fold in range(1, 6):
                for snapshot in ["2012-07-01", "2013-04-01"]:
                    for label in [0, 1]:
                        store = fold * 2 + label
                        rows.append({"repeat_seed": repeat, "architecture": "random_forest",
                            "feature_set": definition, "outer_fold": fold, "StoreID": store,
                            "snapshot": snapshot, "actual_churn": label, "alert": label,
                            "threshold": .5, "status": "selected", "n_orders_3m": 1,
                            "churn_probability": .8 if label else .1})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_pairs_identical_variants_and_is_reproducible():
    data = example_predictions()
    first = study.bootstrap_differences(data, draws=100)
    pd.testing.assert_frame_equal(first, study.bootstrap_differences(data, draws=100))
    assert first[["lower_95", "upper_95", "lower_98_75", "upper_98_75"]].eq(0).all().all()
    data.loc[data.index[0], "actual_churn"] = 1
    with pytest.raises(AssertionError):
        study.bootstrap_differences(data, draws=100)


def test_infeasible_fold_does_not_produce_partial_pooled_operating_metric():
    cases = example_predictions().query("repeat_seed == 42").copy()
    pooled = study.pooled_rows(cases)
    assert pooled.query("subgroup == 'overall'").recall.eq(1).all()
    cases.loc[cases.outer_fold.eq(1) & cases.feature_set.eq("totaldue"), "status"] = "no_feasible_candidate"
    pooled = study.pooled_rows(cases)
    row = pooled[pooled.feature_set.eq("totaldue")].iloc[0]
    assert row.status == "incomplete_inner_selection"
    assert pd.isna(row.recall)
    interval = study.bootstrap_differences(cases, draws=100).iloc[0]
    assert interval.status == "incomplete_inner_selection"


def test_same_recall_higher_precision_requires_nonnegative_uncertainty_and_floor():
    assert study.classify_gain(0, .02, 0, .01, 0, .03, True) == "totaldue_supported_conditionally"
    assert study.classify_gain(0, .02, -.01, -.01, .01, .03, True) == "inconclusive"
    assert study.classify_gain(0, .02, 0, .01, 0, .03, False) == "inconclusive"
    assert study.classify_gain(-.02, .03, -.04, .01, -.01, .04, True) == "tradeoff_not_dominance"


def test_independent_grid_verifier_detects_wrong_confusion_count():
    from verify_nested_revenue_outputs import verify_inner_grid

    rows = []
    for fold in range(1, 6):
        for label, score in [(1, .8), (0, .2)]:
            rows.append({"inner_fold": fold, "actual_churn": label, "churn_probability": score})
    predictions = pd.DataFrame(rows)
    _, grid = study.select_inner(predictions, "test")
    verify_inner_grid(predictions, grid)
    grid.loc[0, "1_tp"] = 0
    with pytest.raises(AssertionError):
        verify_inner_grid(predictions, grid)


def test_duplicate_repeats_do_not_artificially_narrow_cluster_intervals():
    predictions = example_predictions()
    predictions.loc[predictions.feature_set.eq("subtotal") & predictions.StoreID.eq(2), "alert"] = 1
    repeated = study.bootstrap_differences(predictions, draws=100)
    single = study.bootstrap_differences(predictions[predictions.repeat_seed.eq(42)], draws=100)
    pd.testing.assert_frame_equal(repeated, single)

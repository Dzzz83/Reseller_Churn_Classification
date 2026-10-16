"""Synthetic safeguards for descriptive saved-prediction diagnosis."""

import json

import numpy as np
import pandas as pd
import pytest

from comparisons.baseline_diagnosis.analysis import (
    build_cases,
    historical_profiles,
    metric_tables,
    rank_curve,
    ranking_tables,
    stability_table,
    summarize_group,
)
from comparisons.baseline_diagnosis.definitions import POLICIES
from comparisons.baseline_diagnosis.inputs import (
    resolve_study_input,
    validate_partition,
    verify_baseline,
)
from comparisons.baseline_error_diagnosis import run
from reseller_churn.config.frozen_model import FEATURES


@pytest.fixture
def cases():
    features = pd.DataFrame({name: [0.5] * 4 for name in FEATURES})
    features["StoreID"] = [1, 2, 1, 2]
    features["snapshot"] = pd.to_datetime(["2012-07-01"] * 2 + ["2012-10-01"] * 2)
    features["churn"] = [0, 1, 1, 0]
    features["n_orders_3m"] = [1, 1, 1, 0]
    features["weighted_discount_3m"] = [0, 0, 0, np.nan]
    profiles = features[["StoreID", "snapshot"]].copy()
    profiles["n_orders_12m"] = [1, 2, 3, 4]
    profiles["observed_history_span_days"] = [0, 90, 180, 181]
    rows = []
    for repeat, scores in [
        (42, [0.8, 0.8, 0.4, 0.1]),
        (78, [0.9, 0.6, 0.2, 0.3]),
        (88, [0.7, 0.7, 0.05, 0.9]),
    ]:
        for strategy in POLICIES:
            group = features[["StoreID", "snapshot", "churn", "n_orders_3m"]].copy()
            group["repeat"] = repeat
            group["outer_fold"] = 0
            group["score"] = scores
            group["cutoff"] = 0.2 if strategy == "original_baseline" else 0.06
            group["strategy"] = strategy
            rows.append(group)
    return build_cases(features, pd.concat(rows, ignore_index=True), profiles)


def test_tied_scores_have_identical_rank_and_band(cases):
    first = cases[(cases.repeat == 42) & cases.policy.eq("inner_selected")].iloc[:2]
    np.testing.assert_array_equal(first.rank_percentile, [0.25, 0.25])
    assert first.rank_band.nunique() == 1
    assert first.rank_band.iloc[0] == "10-25%"


def test_cutoff_equality_and_policy_invariant_ranking(cases):
    equal = cases[
        (cases.repeat == 78) & cases.policy.eq("inner_selected") & cases.score.eq(0.2)
    ]
    assert equal.predicted_churn.eq(1).all()
    assert equal.result_type.eq("caught_churn").all()
    pivot = cases.pivot(
        index=["StoreID", "snapshot", "repeat"],
        columns="policy",
        values="rank_percentile",
    )
    np.testing.assert_array_equal(pivot.fixed_006, pivot.inner_selected)


def test_profile_boundaries_and_missing_are_not_zero(cases):
    selected = cases[(cases.repeat == 42) & cases.policy.eq("inner_selected")]
    assert selected.history_span.astype(str).tolist() == [
        "0-90 days",
        "0-90 days",
        "91-180 days",
        "181+ days",
    ]
    assert selected.annual_orders.astype(str).tolist() == ["1", "2", "3+", "3+"]
    assert selected.weighted_discount_3m_availability.tolist() == ["available"] * 3 + [
        "missing"
    ]


def test_history_profile_excludes_future_and_old_orders():
    features = pd.DataFrame(
        {"StoreID": [1], "snapshot": pd.to_datetime(["2013-04-01"])}
    )
    orders = pd.DataFrame(
        {
            "StoreID": [1] * 5,
            "SalesOrderID": range(5),
            "OrderDate": pd.to_datetime(
                ["2012-03-31", "2012-04-01", "2013-03-31", "2013-04-01", "2013-05-01"]
            ),
        }
    )
    before = historical_profiles(features, orders)
    assert before.n_orders_12m.item() == 2
    assert before.observed_history_span_days.item() == 364
    pd.testing.assert_frame_equal(
        before, historical_profiles(features, orders.iloc[:3])
    )


def test_metrics_use_saved_decisions_not_one_pooled_cutoff(cases):
    sample = cases.iloc[:4].copy()
    sample["predicted_churn"] = [1, 1, 0, 0]
    metrics = summarize_group(sample)
    assert [metrics[name] for name in ["tp", "fp", "fn", "tn"]] == [1, 1, 1, 1]
    assert (
        metrics["recall"]
        == metrics["precision"]
        == metrics["false_positive_rate"]
        == 0.5
    )


def test_empty_and_single_class_subgroups_are_explicit(cases):
    empty = summarize_group(cases.iloc[:0])
    assert empty["rows"] == 0 and empty["recall"] is None and empty["small_sample"]
    negatives = summarize_group(cases[cases.churn.eq(0)])
    assert negatives["recall"] is None
    assert negatives["average_precision"] is None and negatives["roc_auc"] is None
    metrics, _ = metric_tables(cases)
    empty_slices = metrics[metrics.rows.eq(0)]
    assert not empty_slices.empty and empty_slices.recall.isna().all()


def test_recall_curve_never_splits_a_tie():
    frame = pd.DataFrame({"score": [0.9, 0.9, 0.1], "churn": [0, 1, 1]})
    curve = rank_curve(frame)
    assert len(curve) == 2
    np.testing.assert_array_equal(curve.tp, [1, 2])
    np.testing.assert_array_equal(curve.fp, [1, 1])
    np.testing.assert_array_equal(curve.recall, [0.5, 1])
    pd.testing.assert_frame_equal(curve, rank_curve(frame.iloc[::-1]))


def test_high_recall_burden_is_descriptive_and_has_no_selected_cutoff(cases):
    curves, burden = ranking_tables(cases)
    assert set(burden.target_recall) == {0.9, 0.95, 1}
    assert burden.diagnostic_only.all()
    assert burden.achieved_recall.ge(burden.target_recall).all()
    assert "cutoff" not in burden and "score_boundary" not in burden
    assert set(curves.scope) == {"outer_fold", "repeat_pooled_mixed_models"}


def test_stability_keeps_snapshots_separate_and_does_not_triple_accounts(cases):
    stability = stability_table(cases)
    assert len(stability) == 8  # Four distinct cases, two policies; not 24 predictions.
    assert stability.repetitions.eq(3).all()
    first = stability[
        (stability.StoreID == 1) & stability.snapshot.eq(pd.Timestamp("2012-07-01"))
    ]
    assert first.false_alert_repetitions.eq(3).all()
    assert first.error_pattern.eq("consistent_error").all()
    later = stability[
        (stability.StoreID == 1) & stability.snapshot.eq(pd.Timestamp("2012-10-01"))
    ]
    assert later.missed_churn_repetitions.eq(1).all()
    assert later.error_pattern.eq("occasional_error").all()


@pytest.mark.parametrize(
    "validation_rows, message", [([1, 3], "Reseller leakage"), ([2], "partition")]
)
def test_outer_partition_rejects_leakage_and_missing_rows(validation_rows, message):
    features = pd.DataFrame({"StoreID": [1, 1, 2, 2]})
    training_rows = [0, 2] if len(validation_rows) == 2 else [0, 1]
    split = dict(
        train_rows=training_rows,
        validation_rows=validation_rows,
        train_store_ids=[1, 2],
        validation_store_ids=[1, 2],
    )
    with pytest.raises(ValueError, match=message):
        validate_partition(features, split)


def test_changed_protocol_is_rejected_before_other_inputs_are_read(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"outputs": {}, "protocol_sha256": "0" * 64})
    )
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed study protocol"):
        verify_baseline(tmp_path, tmp_path / "missing-orders.csv")


def test_existing_output_is_preserved_before_reading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "evidence.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        run(tmp_path / "missing-study", tmp_path / "missing-orders", output)
    assert marker.read_text() == "preserve"


def test_moved_study_inputs_resolve_without_rewriting_windows_provenance(tmp_path):
    source = tmp_path / "results/repeated-grouped-precision"
    relocated = (
        source.parent / "development-precision-diagnosis/development_features.csv"
    )
    relocated.parent.mkdir(parents=True)
    relocated.write_text("original bytes")
    recorded = r"C:\another-machine\repo\results\development-precision-diagnosis\development_features.csv"
    assert resolve_study_input(source, recorded) == relocated
    assert relocated.read_text() == "original bytes"


def test_moved_study_input_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe recorded"):
        resolve_study_input(
            tmp_path / "results/study", r"C:\old\results\..\outside.csv"
        )

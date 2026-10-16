"""Shared-fold, ensemble, and reporting safeguards for revenue comparisons."""

import numpy as np
import pandas as pd
import pytest

import compare_subtotal_totaldue as experiment
from test_compare_seven_ten_features import synthetic_data


def test_shared_folds_cover_each_case_and_keep_resellers_together():
    data = synthetic_data().sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    first = experiment.grouped_folds(data)
    repeated = experiment.grouped_folds(data)
    held_out = []
    for (fold, train, validation), (_, other_train, other_validation) in zip(first, repeated):
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(validation, other_validation)
        assert set(data.iloc[train].StoreID).isdisjoint(data.iloc[validation].StoreID)
        held_out.extend(validation)
    assert sorted(held_out) == list(range(len(data)))
    membership = experiment.fold_membership(data, first)
    assert membership[membership.role.eq("validation")].groupby("StoreID").fold.nunique().eq(1).all()


def seed_predictions():
    rows = []
    for definition in ["subtotal", "totaldue"]:
        for index, seed in enumerate(experiment.MODEL_SEEDS):
            for store, label, score in [(1, 1, .7), (2, 0, .1)]:
                rows.append({"architecture": "random_forest", "feature_set": definition,
                             "fold": "fold_1", "StoreID": store, "snapshot": "2013-01-01",
                             "seed": seed, "actual_churn": label, "n_orders_3m": 1,
                             "churn_probability": score + .01 * index})
    return pd.DataFrame(rows)


def test_seed_average_and_missing_or_duplicate_seed_rejection():
    individual = seed_predictions()
    ensemble = experiment.average_seeds(individual)
    assert ensemble[ensemble.StoreID.eq(1)].churn_probability.tolist() == pytest.approx([.72, .72])
    with pytest.raises(ValueError, match="Incomplete seed"):
        experiment.average_seeds(individual.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate seed"):
        experiment.average_seeds(pd.concat([individual, individual.iloc[:1]]))


def test_seed_labels_must_agree():
    individual = seed_predictions()
    individual.loc[0, "actual_churn"] = 0
    with pytest.raises(ValueError, match="disagree"):
        experiment.average_seeds(individual)


def test_selection_reports_pooled_counts_and_seed_cutoffs():
    individual = seed_predictions()
    ensemble = experiment.average_seeds(individual)
    selections, summary, grid = experiment.summarize(ensemble)
    assert summary.status.eq("selected").all()
    assert len(grid) == 202
    metrics = experiment.metrics_at_cutoffs(ensemble, selections)
    pooled = metrics[metrics.fold.eq("pooled") & metrics.subgroup.eq("overall")]
    assert pooled.tp.tolist() == [1, 1]
    assert pooled.fp.tolist() == [0, 0]
    for _, seed_cases in individual.groupby("seed"):
        seed_metrics = experiment.metrics_at_cutoffs(seed_cases, selections)
        for definition, group in seed_metrics.groupby("feature_set"):
            assert group.threshold.eq(selections[definition]["threshold"]).all()
    assert experiment.matched_cases(ensemble, selections).change_type.eq("unchanged").all()


def test_infeasible_selections_have_no_operating_comparison():
    ensemble = experiment.average_seeds(seed_predictions())
    ensemble["actual_churn"] = 0
    selections, summary, _ = experiment.summarize(ensemble)
    assert summary.status.eq("no_feasible_candidate").all()
    assert experiment.matched_cases(ensemble, selections).empty
    assert experiment.metrics_at_cutoffs(ensemble, selections).empty


def test_baseline_check_rejects_changed_probabilities():
    current = experiment.average_seeds(seed_predictions())
    boosting = current.copy()
    boosting["architecture"] = "hist_gradient_boosting"
    current = pd.concat([current, boosting], ignore_index=True)
    saved = current[current.feature_set.eq("subtotal")].copy()
    saved["feature_set"] = saved.architecture.map(experiment.CONFIGURATIONS)
    saved["validation_method"] = "grouped_kfold"
    assert len(experiment.verify_baseline(current, saved)) == 2
    saved.loc[saved.index[0], "churn_probability"] += .01
    with pytest.raises(AssertionError):
        experiment.verify_baseline(current, saved)


def test_existing_run_rejected_before_reading_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    (tmp_path / "results/debugging/subtotal_vs_totaldue/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Choose a new run ID"):
        experiment.run(tmp_path / "absent", "existing")


def test_input_hash_detects_modified_preparation(tmp_path):
    import json

    path = tmp_path / "development_subtotal.csv"
    path.write_text("modified input\n")
    (tmp_path / "preparation_manifest.json").write_text(json.dumps({
        "output_sha256": {path.name: "incorrect-hash"},
    }))
    with pytest.raises(ValueError, match="artifact hash differs"):
        experiment.load_pair(tmp_path)


def test_saved_report_verifier_detects_changed_metric():
    from verify_subtotal_totaldue_outputs import assert_report_equal

    expected = pd.DataFrame({"feature_set": ["subtotal"], "recall": [.5]})
    observed = expected.copy()
    observed.loc[0, "recall"] = .6
    with pytest.raises(AssertionError):
        assert_report_equal(observed, expected, ["feature_set"])

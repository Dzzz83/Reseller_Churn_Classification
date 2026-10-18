import json

import numpy as np
import pandas as pd
import pytest

from reseller_churn.modeling import grouped_development as development
from reseller_churn.modeling.bundles import load_grouped_model
from reseller_churn.modeling.candidates import Candidate
from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
)


def test_outer_metrics_use_fold_specific_retained_decisions():
    predictions = pd.DataFrame(
        {
            "snapshot": pd.to_datetime(["2013-01-01", "2013-01-01"]),
            "repeat": [42, 42],
            "policy": [0, 0],
            "outer_fold": [0, 1],
            "churn": [1, 0],
            "churn_probability": [0.4, 0.7],
            "threshold": [0.3, 0.8],
            "predicted_churn": [1, 0],
        }
    )
    overall = development._assess_outer_predictions(predictions, 2).iloc[0]
    assert overall.tp == 1 and overall.tn == 1 and overall.fp == 0 and overall.fn == 0
    assert overall.complete


@pytest.mark.parametrize(
    "architecture,parameters",
    [
        ("logistic_regression", {"C": 1.0, "max_iter": 500}),
        (
            "random_forest",
            {
                "n_estimators": 3,
                "max_depth": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        ),
    ],
)
def test_other_registered_architectures_reload_after_grouped_selection(
    tmp_path,
    grouped_data,
    architecture,
    parameters,
):
    candidate = Candidate(architecture, "ten", "balanced_weights", parameters, (42,))
    output = tmp_path / "study"
    development.run_grouped_study(
        grouped_data,
        [candidate],
        output,
        tmp_path / "models",
        plan=GroupedValidationPlan(2, 2, (42,)),
        policies=[ThresholdPolicy(minimum_precision=0.1)],
    )
    model, _ = load_grouped_model(output / "selection/policy_00.json")
    assert np.isfinite(model.probabilities(grouped_data)).all()


def test_nested_study_saves_complete_groups_and_reloads_models(
    tmp_path, grouped_data, small_candidates
):
    output = tmp_path / "development"
    plan = GroupedValidationPlan(3, 2, (42,), 78)
    development.run_grouped_study(
        grouped_data,
        small_candidates,
        output,
        tmp_path / "models",
        plan=plan,
        policies=[
            ThresholdPolicy(minimum_precision=0.1),
            ThresholdPolicy(minimum_precision=0.2),
        ],
    )
    predictions = pd.read_csv(output / "validation_predictions.csv")
    assert len(predictions) == len(grouped_data) * 2
    assert not predictions.duplicated(["repeat", "policy", "StoreID", "snapshot"]).any()
    assert predictions.groupby(["policy", "StoreID"]).outer_fold.nunique().eq(1).all()
    assert predictions.predicted_churn.eq(
        predictions.churn_probability.ge(predictions.threshold)
    ).all()
    for folder in sorted(output.glob("repeat_*/fold_*")):
        split = json.loads((folder / "split.json").read_text())
        assert set(split["training_store_ids"]).isdisjoint(
            split["validation_store_ids"]
        )
        inner = pd.read_csv(folder / "inner_oof.csv")
        assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()
        assert set(inner.StoreID).isdisjoint(split["validation_store_ids"])
    selection = output / "selection/policy_00.json"
    model, manifest = load_grouped_model(selection)
    assert manifest["validation_plan"]["design"].startswith("repeated_nested")
    assert np.isfinite(model.probabilities(grouped_data)).all()
    with pytest.raises(FileExistsError):
        development.run_grouped_study(
            grouped_data, small_candidates, output, tmp_path / "other"
        )
    selection.write_text(selection.read_text() + " ")
    with pytest.raises(ValueError, match="Selection checksum"):
        load_grouped_model(selection)


def test_infeasible_partition_fails_before_fitting_or_output(
    tmp_path, grouped_data, small_candidates, monkeypatch
):
    # Only one positive reseller remains; some grouped training partition lacks a class.
    grouped_data["churn"] = grouped_data.StoreID.eq(1).astype(int)
    monkeypatch.setattr(
        development, "fit_and_predict_candidate", lambda *a: pytest.fail("must not fit")
    )
    output = tmp_path / "development"
    with pytest.raises(ValueError, match="both classes|Both fitting outcomes"):
        development.run_grouped_study(
            grouped_data,
            small_candidates,
            output,
            tmp_path / "models",
            plan=GroupedValidationPlan(2, 2, (42,)),
        )
    assert not output.exists()


def test_inner_fit_sees_only_training_groups_and_unlabeled_features(
    grouped_data, small_candidates, monkeypatch
):
    training_sets = []

    def recorded_fit(candidate, training_features, target, validation_features):
        assert "churn" not in validation_features
        assert list(validation_features) == list(candidate.features)
        np.testing.assert_array_equal(target, training_features.churn)
        training_sets.append(set(training_features.StoreID))
        return np.full(len(validation_features), 0.25), ()

    monkeypatch.setattr(development, "fit_and_predict_candidate", recorded_fit)
    plan = GroupedValidationPlan(3, 2, (42,), 78)
    scores, assignments = development.inner_oof_scores(
        grouped_data, small_candidates[:1], plan
    )
    for fold, training_ids in enumerate(training_sets):
        assert training_ids.isdisjoint(grouped_data.loc[assignments == fold, "StoreID"])
    assert np.isfinite(next(iter(scores.values()))).all()


def test_outer_labels_cannot_change_choices_at_fixed_memberships(
    tmp_path,
    grouped_data,
    small_candidates,
    monkeypatch,
):
    plan = GroupedValidationPlan(2, 2, (42,))
    split = next(plan.outer_splits(grouped_data))
    outer_training_stores = set(grouped_data.iloc[split[2]].StoreID)

    def recorded_fit(candidate, training, target, features):
        assert set(training.StoreID).issubset(outer_training_stores)
        assert not {"StoreID", "snapshot", "churn"}.intersection(features)
        return np.full(len(features), 0.2), ()

    monkeypatch.setattr(development, "fit_and_predict_candidate", recorded_fit)
    policies = [ThresholdPolicy(minimum_precision=0.1)]
    _, first = development._fit_outer_fold(
        grouped_data, small_candidates, tmp_path / "first", plan, policies, split
    )
    changed = grouped_data.copy()
    changed.loc[split[3], "churn"] = 1 - changed.loc[split[3], "churn"]
    _, second = development._fit_outer_fold(
        changed, small_candidates, tmp_path / "second", plan, policies, split
    )
    assert first == second


def test_infeasible_policies_leave_readable_empty_reports_and_no_refit(
    tmp_path,
    grouped_data,
    small_candidates,
    monkeypatch,
):
    monkeypatch.setattr(
        development,
        "fit_and_predict_candidate",
        lambda candidate, train, target, valid: (np.full(len(valid), 0.2), ()),
    )
    monkeypatch.setattr(
        development,
        "fit_candidate_ensemble",
        lambda *a: pytest.fail("infeasible refit"),
    )
    output = tmp_path / "study"
    development.run_grouped_study(
        grouped_data,
        small_candidates,
        output,
        tmp_path / "models",
        plan=GroupedValidationPlan(2, 2, (42,)),
        policies=[ThresholdPolicy(minimum_precision=1.0)],
    )
    assert pd.read_csv(output / "validation_predictions.csv").empty
    assert pd.read_csv(output / "operating_metrics.csv").empty
    assert (
        pd.read_csv(output / "selection_status.csv")
        .status.eq("no_feasible_candidate")
        .all()
    )
    with pytest.raises(ValueError, match="selected grouped"):
        load_grouped_model(output / "selection/policy_00.json")


@pytest.mark.parametrize(
    "mutation", ["snapshot", "duplicate", "missing", "label", "infinite"]
)
def test_invalid_development_tables_rejected(grouped_data, small_candidates, mutation):
    if mutation == "snapshot":
        grouped_data.loc[0, "snapshot"] = pd.Timestamp("2013-10-01")
    elif mutation == "duplicate":
        grouped_data = pd.concat([grouped_data, grouped_data.iloc[[0]]])
    elif mutation == "missing":
        grouped_data = grouped_data.drop(columns="weighted_discount_3m")
    elif mutation == "label":
        grouped_data["churn"] = 2
    else:
        grouped_data.loc[0, "revenue_12m"] = np.inf
    with pytest.raises(ValueError):
        development.validate_grouped_data(grouped_data, small_candidates)


def test_development_cli_runs_grouped_workflow(tmp_path, grouped_data, monkeypatch):
    from reseller_churn.cli import main
    from reseller_churn.data.temporal_dataset import TemporalDataset

    monkeypatch.setattr(
        TemporalDataset,
        "prepare_fold",
        lambda *args: pytest.fail("temporal workflow invoked"),
    )
    source = tmp_path / "development.csv"
    grouped_data.to_csv(source, index=False)
    output = tmp_path / "output"
    main(
        [
            "develop",
            "--development",
            str(source),
            "--output",
            str(output),
            "--models",
            str(tmp_path / "models"),
            "--outer-folds",
            "2",
            "--inner-folds",
            "2",
            "--repeat-seeds",
            "42",
            "--seeds",
            "42",
            "--precision-floors",
            "0.1",
            "--imbalance-strategies",
            "balanced_weights",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stage"] == "grouped_development"
    assert manifest["validation_plan"]["outer_folds"] == 2

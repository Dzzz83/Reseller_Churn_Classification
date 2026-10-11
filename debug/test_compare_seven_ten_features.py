"""Shared splits, model smoke checks, and operating-point edge cases."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import compare_seven_ten_features as experiment


def synthetic_data():
    rows = []
    for snapshot in ["2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01"]:
        for store in range(40):
            row = {"StoreID": store, "snapshot": snapshot, "churn": store % 2}
            row.update({name: float(store + 1) for name in experiment.FEATURE_SETS["ten"]})
            rows.append(row)
    return pd.DataFrame(rows)


def test_splits_are_reproducible_and_reseller_disjoint():
    data = synthetic_data()
    first = experiment.validation_splits(data)
    second = experiment.validation_splits(data)
    held_out = []
    for (fold, training, validation), (_, _, repeated) in zip(
        first["grouped_kfold"], second["grouped_kfold"]
    ):
        assert set(training.StoreID).isdisjoint(validation.StoreID)
        assert validation.index.tolist() == repeated.index.tolist()
        held_out.extend(validation.index)
    assert sorted(held_out) == list(data.index)
    for _, training, validation in first["temporal"]:
        latest_label_end = pd.to_datetime(training.snapshot).max() + pd.DateOffset(months=6)
        assert latest_label_end <= pd.to_datetime(validation.snapshot).min()


@pytest.mark.parametrize("architecture", experiment.ARCHITECTURES)
def test_models_fit_with_missing_discounts_and_predict_finite_scores(architecture, monkeypatch):
    monkeypatch.setattr(experiment, "BASELINE_RANDOM_FOREST", replace(experiment.BASELINE_RANDOM_FOREST, n_estimators=3))
    monkeypatch.setattr(experiment, "BASELINE_GRADIENT_BOOSTING", replace(experiment.BASELINE_GRADIENT_BOOSTING, max_iter=3))
    data = synthetic_data()
    data.loc[data.StoreID.mod(3).eq(0), "discount_change_3m"] = np.nan
    training = data[data.snapshot.eq("2012-07-01")]
    validation = data[data.snapshot.eq("2013-01-01")]
    scores = experiment.fit_scores(architecture, experiment.FEATURE_SETS["ten"], training, validation, 42)
    assert len(scores) == len(validation)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def example_predictions():
    return pd.DataFrame({
        "feature_set": ["seven", "seven", "ten", "ten"],
        "fold": ["fold_1"] * 4, "StoreID": [1, 2, 1, 2],
        "snapshot": ["2013-01-01"] * 4, "n_orders_3m": [1] * 4,
        "actual_churn": [1, 0, 1, 0], "churn_probability": [.8, .2, .9, .1],
    })


def test_operating_points_pair_identical_populations():
    predictions = example_predictions()
    selections, tables = experiment.summarize(predictions, "temporal", "logistic_regression", "ensemble", "ensemble")
    assert all(value["status"] == "selected" for value in selections.values())
    assert len(experiment.compare_cases(predictions, selections)) == 2
    delta = experiment.feature_deltas(tables[0]).iloc[0]
    assert delta.both_feasible
    assert delta.delta_mean_recall == 0


def test_infeasible_threshold_does_not_create_paired_operating_comparison():
    predictions = example_predictions()
    predictions["actual_churn"] = 0
    selections, tables = experiment.summarize(predictions, "temporal", "logistic_regression", "ensemble", "ensemble")
    assert all(value["status"] == "no_feasible_candidate" for value in selections.values())
    assert experiment.compare_cases(predictions, selections).empty
    delta = experiment.feature_deltas(tables[0]).iloc[0]
    assert not delta.both_feasible
    assert pd.isna(delta.delta_mean_recall)


def test_case_pairing_rejects_a_changed_label():
    predictions = example_predictions()
    selections, _ = experiment.summarize(predictions, "temporal", "logistic_regression", "ensemble", "ensemble")
    predictions.loc[2, "actual_churn"] = 0
    with pytest.raises(ValueError, match="labels differ"):
        experiment.compare_cases(predictions, selections)


def test_logistic_preprocessing_is_fitted_only_on_training_rows(monkeypatch):
    original_factory = experiment.ModelFactory.create_logistic_regression
    created = []

    def capture_model(**kwargs):
        model = original_factory(**kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(experiment.ModelFactory, "create_logistic_regression", capture_model)
    data = synthetic_data()
    training = data[data.snapshot.eq("2012-07-01")].copy()
    validation = data[data.snapshot.eq("2013-01-01")].copy()
    training.loc[training.StoreID.mod(3).eq(0), "discount_change_3m"] = np.nan
    validation["discount_change_3m"] = 10000.
    experiment.fit_scores("logistic_regression", experiment.FEATURE_SETS["ten"], training, validation, 42)
    imputer = created[0].named_steps["imputer"]
    index = experiment.FEATURE_SETS["ten"].index("discount_change_3m")
    assert imputer.statistics_[index] == training.discount_change_3m.median()
    assert created[0].named_steps["scaler"].n_samples_seen_ == len(training)

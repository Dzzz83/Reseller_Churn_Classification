"""Synthetic retained-model smoke and prediction-checkpoint stage guards."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from comparisons import grouped_precision_recall as study
from comparisons.crossfit_percentile import FEATURES
from comparisons.positive_floor_october import (
    checked_stage,
    fit_retained,
    predict_retained,
)


def test_retained_synthetic_smoke_disjoint_reference_and_reloaded_scores():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.random((120, len(FEATURES))), columns=FEATURES)
    train["StoreID"] = np.arange(120) + 1
    train["snapshot"] = pd.Timestamp("2012-07-01")
    train["churn"] = np.tile([0, 1], 60)
    held = train.drop(columns="churn").iloc[:6].copy()
    held["StoreID"] += 1000
    score, records, retained = fit_retained(train, held)
    assert len(records) == 10 and len(retained) == 10
    np.testing.assert_array_equal(score, predict_retained(retained, held))
    for record, pair in zip(records, retained):
        fitting = {r["StoreID"] for r in record["fitting_cases"]}
        reference = {r["StoreID"] for r in record["reference_cases"]}
        assert fitting.isdisjoint(reference) and fitting | reference == set(
            train.StoreID
        )
        assert fitting.isdisjoint(held.StoreID) and reference.isdisjoint(held.StoreID)
        assert list(pair["model"].feature_names_in_) == FEATURES
        raw = np.array(record["raw_held_scores"])
        mapped = expit(
            record["slope"] * logit(np.clip(raw, 1e-12, 1 - 1e-12))
            + record["intercept"]
        )
        np.testing.assert_array_equal(mapped, record["mapped_scores"])
        assert record["seed"] in study.MODEL_SEEDS


def test_outcome_stage_requires_existing_prediction_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        checked_stage(tmp_path)


def test_outcome_join_normalizes_datetime_predictions_without_changing_cases():
    from comparisons.positive_floor_october_evaluation import join_supplied_outcomes

    predictions = pd.DataFrame(
        dict(
            StoreID=[1, 1],
            snapshot=pd.to_datetime(["2013-10-01"] * 2),
            strategy=["baseline", "positive_floor_veto"],
            score=[0.7, 0.0],
            cutoff=[0.5, 0.5],
        )
    )
    labels = pd.DataFrame(dict(StoreID=[1], snapshot=["2013-10-01"], churn=[1]))
    joined = join_supplied_outcomes(predictions, labels, expected_rows=1)
    assert len(joined) == 2 and joined.churn.eq(1).all()
    np.testing.assert_array_equal(joined.score, predictions.score)
    np.testing.assert_array_equal(joined.cutoff, predictions.cutoff)


def test_no_fit_outcome_stage_synthetic_smoke(tmp_path, monkeypatch):
    from comparisons import positive_floor_october_evaluation as evaluation
    from reseller_churn.artifacts import verify_manifest, write_manifest

    source = tmp_path / "results/current-model-diagnosis"
    source.mkdir(parents=True)
    labels = pd.DataFrame(
        dict(
            StoreID=np.arange(492) + 1,
            snapshot="2013-10-01",
            churn=np.tile([0, 1], 246),
        )
    )
    labels.to_csv(source / "october_cases.csv", index=False)
    write_manifest(source, dict(stage="synthetic"))
    output = tmp_path / "prepared"
    output.mkdir()
    base = labels.drop(columns="churn").assign(
        n_orders_3m=np.tile([0, 0, 1, 1], 123), score=0.7, cutoff=0.5, predicted_churn=1
    )
    predictions = pd.concat(
        [
            base.assign(strategy=name)
            for name in ["baseline", "positive_floor_veto", "historical_0.06"]
        ],
        ignore_index=True,
    )
    predictions.to_csv(output / "unlabeled_predictions.csv", index=False)
    write_manifest(output, dict(stage="synthetic_prediction_checkpoint"))
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "checked_stage", lambda path: None)
    evaluation.evaluate(output)
    verify_manifest(output / "completed_evaluation")
    metrics = pd.read_csv(output / "completed_evaluation/metrics.csv")
    overall = metrics[metrics.subgroup.eq("overall")]
    assert len(metrics) == 9 and overall.tp.eq(246).all() and overall.fp.eq(246).all()
    assert overall.recall.eq(1).all() and overall.precision.eq(0.5).all()

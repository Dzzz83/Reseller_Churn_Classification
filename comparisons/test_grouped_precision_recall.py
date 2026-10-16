"""Synthetic tests for grouped selection and outcome-isolation safeguards."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from comparisons import grouped_precision_recall as study


def test_protected_snapshot_and_duplicate_population_rejected():
    frame = study.synthetic_development()
    frame.loc[0, "snapshot"] = pd.Timestamp("2013-10-01")
    with pytest.raises(ValueError, match="approved development"):
        study.validate_development(frame, canonical=False)
    frame = study.synthetic_development()
    with pytest.raises(ValueError, match="duplicate"):
        study.validate_development(pd.concat([frame, frame.iloc[[0]]]), canonical=False)


def test_all_snapshots_of_a_store_stay_together_at_both_levels():
    frame = study.synthetic_development()
    for repeat in study.REPEAT_SEEDS:
        for ti, vi in study.grouped_splits(frame, 5, repeat):
            train = frame.iloc[ti]
            assert set(train.StoreID).isdisjoint(frame.iloc[vi].StoreID)
            for it, iv in study.grouped_splits(train, 3, 78):
                assert set(train.iloc[it].StoreID).isdisjoint(train.iloc[iv].StoreID)


def test_threshold_is_strictly_above_90_percent_and_includes_equality():
    labels = np.r_[np.ones(10), np.zeros(2)]
    scores = np.r_[np.linspace(0.1, 1.0, 10), [0.2, 0.05]]
    selected = study.select_threshold(labels, scores)
    assert selected["cutoff"] == pytest.approx(0.1)
    assert selected["recall"] == 1.0
    assert study.decision_metrics([1], [0.1], 0.1)["tp"] == 1


def test_feature_selection_uses_precision_then_recall_then_simplicity():
    rows = [
        dict(candidate="a", precision=0.4, recall=0.91, feature_count=16),
        dict(candidate="b", precision=0.4, recall=0.92, feature_count=16),
        dict(candidate="c", precision=0.4, recall=0.92, feature_count=10),
        dict(candidate="d", precision=0.8, recall=0.90, feature_count=10),
    ]
    assert study.choose_configuration(rows)["candidate"] == "c"


def test_fitting_uses_only_training_weights(monkeypatch):
    frame = study.synthetic_development()
    train, valid = frame.iloc[:200], frame.iloc[200:]
    seen = []

    class Estimator:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target, sample_weight):
            np.testing.assert_array_equal(target, train.churn)
            np.testing.assert_array_equal(
                sample_weight, compute_sample_weight("balanced", train.churn)
            )
            seen.append(self.parameters["random_state"])

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    monkeypatch.setattr(study, "HistGradientBoostingClassifier", Estimator)
    monkeypatch.setattr(study.joblib, "hash", lambda model: "synthetic-estimator")
    study.fit_scores(
        train, valid[study.FEATURES], study.candidates()[0], seeds=(42, 78)
    )
    assert seen == [42, 78]


def test_outer_label_changes_cannot_change_selected_configuration(
    tmp_path, monkeypatch
):
    data = study.synthetic_development()
    ti, vi = study.grouped_splits(data, 2, 42)[0]
    training_stores = set(data.iloc[ti].StoreID)

    def fake_scores(train, validation_features, candidate, seeds, retain_models=False):
        assert set(train.StoreID).issubset(training_stores)
        assert "churn" not in validation_features
        return np.full(len(validation_features), 0.2), [], 0.0, []

    monkeypatch.setattr(study, "fit_scores", fake_scores)
    grid = [study.candidates()[0], study.candidates()[-1]]
    study.evaluate_outer_fold(data, 42, 0, ti, vi, tmp_path / "first", grid, 2, (42,))
    changed = data.copy()
    changed.loc[vi, "churn"] = 1 - changed.loc[vi, "churn"]
    study.evaluate_outer_fold(
        changed, 42, 0, ti, vi, tmp_path / "second", grid, 2, (42,)
    )
    first = json.loads((tmp_path / "first/selected.json").read_text())
    second = json.loads((tmp_path / "second/selected.json").read_text())
    assert first == second


def test_cluster_bootstrap_pairs_all_repeats_and_snapshots():
    rows = []
    for repeat in (42, 78):
        for store in range(1, 5):
            for snapshot in study.SNAPSHOTS:
                for strategy in ("original_baseline", "adaptive_selection"):
                    rows.append(
                        dict(
                            repeat=repeat,
                            StoreID=store,
                            snapshot=snapshot,
                            churn=store % 2,
                            score=0.8 if store % 2 else 0.1,
                            cutoff=0.5,
                            strategy=strategy,
                        )
                    )
    result = study.cluster_bootstrap(pd.DataFrame(rows), draws=100)
    assert result["precision_delta_95"] == [0.0, 0.0]
    assert result["recall_delta_95"] == [0.0, 0.0]


def test_csv_scores_round_trip_at_decision_boundary(tmp_path):
    score = 0.12345678901234568
    path = tmp_path / "predictions.csv"
    pd.DataFrame(dict(score=[score], cutoff=[score])).to_csv(path, index=False)
    loaded = study.read_csv(path)
    assert loaded.score.item() == score
    assert loaded.score.ge(loaded.cutoff).item()

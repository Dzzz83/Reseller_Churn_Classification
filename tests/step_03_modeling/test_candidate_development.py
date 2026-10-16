import json

import joblib
import numpy as np
import pandas as pd
import pytest

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.model_settings import (
    GradientBoostingSettings,
    RandomForestSettings,
)
from reseller_churn.config.validation_settings import (
    DEVELOPMENT_FOLDS,
    LABELED_SNAPSHOTS,
)
from reseller_churn.evaluation.selection import SELECTION_RULE
from reseller_churn.modeling.candidates import Candidate, candidate_from_dict
from reseller_churn.modeling.development import (
    FittedProbabilityEnsemble,
    dataframe_sha256,
    fit_and_predict_candidate,
    fold_manifest,
    load_selected_predictor,
    refit_selected_candidate,
    validate_labeled_development_data,
)
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.seeded_predictions import SeededProbabilityPredictor


def make_labeled_development_data() -> pd.DataFrame:
    all_features = sorted({feature for features in FEATURE_SETS.values() for feature in features})
    rows = []
    for snapshot_index, snapshot in enumerate(LABELED_SNAPSHOTS):
        for store_id in range(1, 9):
            row = {
                "StoreID": store_id,
                "snapshot": snapshot,
                "churn": int(store_id in {1, 2}),
            }
            row.update(
                {
                    feature: float(store_id + snapshot_index)
                    for feature in all_features
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_candidate() -> Candidate:
    return Candidate(
        architecture="logistic_regression",
        feature_set="rfm",
        imbalance_strategy="none",
        parameters={"C": 1.0, "max_iter": 2000},
        seeds=(42, 78),
    )


def make_selected_manifest(candidate: Candidate, data_hash: str) -> dict:
    return {
        "status": "selected",
        "candidate": candidate.to_dict(),
        "threshold": 0.5,
        "minimum_precision": 0.1,
        "selection_rule": SELECTION_RULE,
        "approved_training_snapshots": list(LABELED_SNAPSHOTS),
        "folds": fold_manifest(),
        "data_sha256": data_hash,
        "prediction_policy": {
            "seed_aggregation": "mean_predicted_probability",
            "seeds": list(candidate.seeds),
            "threshold_comparison": ">=",
        },
        "selected_fold_metrics": {
            fold.name: {"precision": 0.5} for fold in DEVELOPMENT_FOLDS
        },
    }


def test_candidate_id_and_manifest_features_are_verified():
    candidate = make_candidate()
    manifest_value = candidate.to_dict()
    assert candidate_from_dict(manifest_value).candidate_id == candidate.candidate_id

    manifest_value["features"] = ["churn"]
    with pytest.raises(ValueError, match="do not match"):
        candidate_from_dict(manifest_value)


@pytest.mark.parametrize(
    "extra_snapshot",
    ["2013-07-01", "2013-10-01"],
)
def test_development_data_rejects_rows_outside_approved_snapshots(extra_snapshot):
    data = make_labeled_development_data()
    extra_row = data.iloc[[0]].copy()
    extra_row["snapshot"] = extra_snapshot

    with pytest.raises(ValueError, match="outside|Protected final-test"):
        validate_labeled_development_data(pd.concat([data, extra_row], ignore_index=True))


def test_no_feasible_manifest_cannot_be_refit():
    data = make_labeled_development_data()
    manifest = {"status": "no_feasible_candidate"}

    with pytest.raises(ValueError, match="Only a selected manifest"):
        refit_selected_candidate(data, manifest)


def test_refit_rejects_modified_data_with_the_original_hash():
    data = make_labeled_development_data()
    original_hash = dataframe_sha256(data)
    manifest = make_selected_manifest(make_candidate(), original_hash)
    data.loc[0, "recency_days"] += 1

    with pytest.raises(ValueError, match="does not match the expected hash"):
        refit_selected_candidate(data, manifest, original_hash)


def test_saved_seed_ensemble_reloads_with_the_same_probabilities(
    tmp_path,
    monkeypatch,
):
    import reseller_churn.modeling.development as development

    data = make_labeled_development_data()
    candidate = make_candidate()
    data_hash = dataframe_sha256(data)
    manifest = make_selected_manifest(candidate, data_hash)
    predictor = refit_selected_candidate(data, manifest, data_hash)
    assert isinstance(predictor, FittedProbabilityEnsemble)
    assert len(predictor.models) == len(candidate.seeds)

    inference_rows = data[data["snapshot"] == LABELED_SNAPSHOTS[-1]]
    expected_probabilities = predictor.predict_probabilities(inference_rows)

    project_root = tmp_path
    artifact_path = project_root / "models" / "provisional" / "run-smoke" / "model.joblib"
    artifact_path.parent.mkdir(parents=True)
    joblib.dump(predictor, artifact_path)
    manifest["model_bundle"] = {
        "path": artifact_path.relative_to(project_root).as_posix(),
        "sha256": file_sha256(artifact_path),
    }
    manifest_path = project_root / "results" / "run-smoke" / "selection.json"
    write_json(manifest_path, manifest)

    monkeypatch.setattr(development, "PROJECT_ROOT", project_root)
    reloaded_manifest, reloaded_predictor = load_selected_predictor(manifest_path)

    assert reloaded_manifest["candidate"]["candidate_id"] == candidate.candidate_id
    np.testing.assert_allclose(
        reloaded_predictor.predict_probabilities(inference_rows),
        expected_probabilities,
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        reloaded_predictor.predict(inference_rows, reloaded_manifest["threshold"]),
        predictor.predict(inference_rows, manifest["threshold"]),
    )


def test_manifest_json_roundtrip_preserves_fold_configuration(tmp_path):
    candidate = make_candidate()
    manifest = make_selected_manifest(candidate, dataframe_sha256(make_labeled_development_data()))
    path = tmp_path / "selection.json"
    write_json(path, manifest)

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["folds"] == fold_manifest()


def test_loader_rejects_selected_manifest_without_model_bundle(tmp_path):
    candidate = make_candidate()
    manifest = make_selected_manifest(
        candidate,
        dataframe_sha256(make_labeled_development_data()),
    )
    path = tmp_path / "selection.json"
    write_json(path, manifest)

    with pytest.raises(ValueError, match="does not identify a model bundle"):
        load_selected_predictor(path)


def test_shared_candidate_fitting_matches_legacy_seeded_predictions():
    random = np.random.default_rng(2026)
    training_target = pd.Series([0] * 36 + [1] * 12, name="churn")
    training_features = pd.DataFrame(
        random.normal(size=(len(training_target), len(FEATURE_SETS["rfm"]))),
        columns=FEATURE_SETS["rfm"],
    )
    training_features.loc[training_target == 1, "recency_days"] += 1.5
    validation_features = pd.DataFrame(
        random.normal(size=(10, len(FEATURE_SETS["rfm"]))),
        columns=FEATURE_SETS["rfm"],
    )

    logistic_candidate = Candidate(
        architecture="logistic_regression",
        feature_set="rfm",
        imbalance_strategy="none",
        parameters={"C": 1.0, "max_iter": 2000},
        seeds=(42,),
    )
    shared_logistic, _ = fit_and_predict_candidate(
        logistic_candidate,
        training_features,
        training_target,
        validation_features,
    )
    legacy_logistic = ModelFactory.create_logistic_regression()
    legacy_logistic.fit(training_features, training_target)
    np.testing.assert_allclose(
        shared_logistic,
        legacy_logistic.predict_proba(validation_features)[:, 1],
        rtol=0,
        atol=0,
    )

    random_forest_settings = RandomForestSettings(
        n_estimators=12,
        max_features="sqrt",
    )
    random_forest_candidate = Candidate(
        architecture="random_forest",
        feature_set="rfm",
        imbalance_strategy="none",
        parameters={
            "n_estimators": random_forest_settings.n_estimators,
            "max_depth": random_forest_settings.max_depth,
            "min_samples_leaf": random_forest_settings.min_samples_leaf,
            "max_features": random_forest_settings.max_features,
        },
        seeds=(11, 29),
    )
    shared_random_forest, _ = fit_and_predict_candidate(
        random_forest_candidate,
        training_features,
        training_target,
        validation_features,
    )
    legacy_random_forest = SeededProbabilityPredictor.random_forest_by_seed(
        training_features,
        training_target,
        validation_features,
        random_forest_settings,
        seeds=random_forest_candidate.seeds,
    )
    np.testing.assert_allclose(
        shared_random_forest,
        SeededProbabilityPredictor.average(legacy_random_forest),
        rtol=0,
        atol=0,
    )

    boosting_settings = GradientBoostingSettings(
        max_iter=8,
        max_leaf_nodes=5,
        min_samples_leaf=2,
    )
    boosting_candidate = Candidate(
        architecture="hist_gradient_boosting",
        feature_set="rfm",
        imbalance_strategy="random_oversampling",
        parameters={
            "learning_rate": boosting_settings.learning_rate,
            "max_iter": boosting_settings.max_iter,
            "max_leaf_nodes": boosting_settings.max_leaf_nodes,
            "min_samples_leaf": boosting_settings.min_samples_leaf,
            "l2_regularization": boosting_settings.l2_regularization,
        },
        seeds=(11, 29),
    )
    shared_boosting, _ = fit_and_predict_candidate(
        boosting_candidate,
        training_features,
        training_target,
        validation_features,
    )
    legacy_boosting = SeededProbabilityPredictor.gradient_boosting_by_seed(
        training_features,
        training_target,
        validation_features,
        boosting_settings,
        seeds=boosting_candidate.seeds,
        oversample_training=True,
    )
    np.testing.assert_allclose(
        shared_boosting,
        SeededProbabilityPredictor.average(legacy_boosting),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("architecture", "parameters", "fit_weight_key"),
    [
        (
            "logistic_regression",
            {"C": 1.0, "max_iter": 100},
            "model__sample_weight",
        ),
        (
            "random_forest",
            {
                "n_estimators": 2,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "model__sample_weight",
        ),
        (
            "hist_gradient_boosting",
            {
                "learning_rate": 0.1,
                "max_iter": 2,
                "max_leaf_nodes": 3,
                "min_samples_leaf": 1,
                "l2_regularization": 0.0,
            },
            "sample_weight",
        ),
    ],
)
def test_balanced_weights_use_training_rows_and_leave_validation_untouched(
    architecture,
    parameters,
    fit_weight_key,
    monkeypatch,
):
    observations = []

    class RecordingModel:
        def fit(self, features, target, **fit_kwargs):
            observations.append(
                {
                    "training_features": features.copy(),
                    "training_target": pd.Series(target).copy(),
                    "fit_kwargs": fit_kwargs,
                }
            )

        def predict_proba(self, features):
            observations[-1]["validation_features"] = features.copy()
            return np.tile([0.75, 0.25], (len(features), 1))

    monkeypatch.setattr(
        ModelFactory,
        "create_logistic_regression",
        lambda **kwargs: RecordingModel(),
    )
    monkeypatch.setattr(
        ModelFactory,
        "create_random_forest",
        lambda settings, random_seed: RecordingModel(),
    )
    monkeypatch.setattr(
        ModelFactory,
        "create_gradient_boosting",
        lambda settings, random_seed: RecordingModel(),
    )

    features = pd.DataFrame(
        {
            "recency_days": [1.0, 2.0, 3.0, 4.0],
            "n_orders_12m": [4.0, 3.0, 2.0, 1.0],
            "revenue_12m": [100.0, 200.0, 300.0, 400.0],
        }
    )
    target = pd.Series([0, 0, 0, 1], name="churn")
    validation_features = pd.DataFrame(
        {
            "recency_days": [1000.0, 2000.0],
            "n_orders_12m": [3000.0, 4000.0],
            "revenue_12m": [5000.0, 6000.0],
        }
    )
    candidate = Candidate(
        architecture=architecture,
        feature_set="rfm",
        imbalance_strategy="balanced_weights",
        parameters=parameters,
        seeds=(42,),
    )

    probabilities, _ = fit_and_predict_candidate(
        candidate,
        features,
        target,
        validation_features,
    )

    assert len(observations) == 1
    observed = observations[0]
    pd.testing.assert_frame_equal(observed["training_features"], features)
    pd.testing.assert_series_equal(observed["training_target"], target)
    pd.testing.assert_frame_equal(
        observed["validation_features"], validation_features
    )
    weights = observed["fit_kwargs"][fit_weight_key]
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2.0])
    np.testing.assert_allclose(probabilities, [0.25, 0.25])

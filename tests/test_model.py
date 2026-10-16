import numpy as np
import pandas as pd
import pytest

from reseller_churn.config import FEATURES, SEEDS
from reseller_churn.data import build_development, build_features, load_sources
from reseller_churn.model import (
    FixedEnsemble,
    load_model,
    predict,
    ranked_predictions,
    train,
)


def training_data(source_paths):
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    return build_development(source), build_features(source, "2013-10-01")


def test_fixed_training_reload_and_ranking(source_paths, tmp_path):
    development, features = training_data(source_paths)
    development.to_csv(tmp_path / "development.csv", index=False)
    train(tmp_path / "development.csv", tmp_path / "model", "2013-10-01")
    ensemble, manifest = load_model(tmp_path / "model")
    assert len(ensemble.models) == len(SEEDS)
    expected = np.mean(
        [
            model.predict_proba(features[list(FEATURES)])[:, 1]
            for model in ensemble.models
        ],
        axis=0,
    )
    np.testing.assert_allclose(expected, ensemble.probabilities(features))
    ranked = ranked_predictions(ensemble, features)
    assert ranked.churn_probability.is_monotonic_decreasing
    assert ranked["rank"].tolist() == list(range(1, len(features) + 1))
    features.to_csv(tmp_path / "features.csv", index=False)
    predict(tmp_path / "model", tmp_path / "features.csv", tmp_path / "prediction")
    assert manifest["training_churners"] == int(development.churn.sum())
    with pytest.raises(FileExistsError):
        predict(tmp_path / "model", tmp_path / "features.csv", tmp_path / "prediction")
    model_path = tmp_path / "model/ensemble.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path / "model")


def test_each_seed_uses_training_class_weights(source_paths, tmp_path, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    development, _ = training_data(source_paths)
    development.to_csv(tmp_path / "development.csv", index=False)
    expected = development.sort_values(["snapshot", "StoreID"])
    expected_weights = compute_sample_weight("balanced", expected.churn)
    original_fit = HistGradientBoostingClassifier.fit
    observed_seeds = []

    def checked_fit(model, features, target, sample_weight=None):
        observed_seeds.append(model.random_state)
        np.testing.assert_array_equal(target.to_numpy(), expected.churn.to_numpy())
        np.testing.assert_allclose(sample_weight, expected_weights)
        return original_fit(model, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", checked_fit)
    train(tmp_path / "development.csv", tmp_path / "model", "2013-10-01")
    assert observed_seeds == list(SEEDS)


def test_ties_sorted_by_store_id(source_paths):
    _, features = training_data(source_paths)

    class ConstantModel:
        def predict_proba(self, frame):
            return np.full((len(frame), 2), 0.5)

    ensemble = FixedEnsemble((ConstantModel(),), {})
    ranked = ranked_predictions(ensemble, features.iloc[::-1])
    assert ranked.StoreID.tolist() == sorted(features.StoreID)


def test_scoring_csv_preserves_a_float_at_a_model_boundary(
    source_paths, tmp_path, monkeypatch
):
    import reseller_churn.model as model_module

    _, features = training_data(source_paths)
    boundary = 0.12345678901234568
    features["weighted_discount_3m"] = boundary
    features.to_csv(tmp_path / "features.csv", index=False)

    class BoundaryModel:
        def predict_proba(self, frame):
            positive = frame.weighted_discount_3m.ge(boundary).astype(float).to_numpy()
            return np.column_stack([1 - positive, positive])

    ensemble = FixedEnsemble((BoundaryModel(),), {})
    expected = ensemble.probabilities(features)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        model_module,
        "load_model",
        lambda _: (
            ensemble,
            {
                "prediction_date": "2013-10-01",
                "configuration": {},
            },
        ),
    )
    predict(model_directory, tmp_path / "features.csv", tmp_path / "prediction")
    actual = pd.read_csv(tmp_path / "prediction/ranked_predictions.csv")
    np.testing.assert_array_equal(actual.churn_probability, expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "protected",
        "premature",
        "nonbinary",
        "missing_feature",
        "infinite_feature",
        "duplicate",
    ],
)
def test_training_rejects_ineligible_inputs(source_paths, tmp_path, mutation):
    development, _ = training_data(source_paths)
    prediction_date = "2013-10-01"
    if mutation == "protected":
        development.loc[0, "snapshot"] = pd.Timestamp("2013-10-01")
    elif mutation == "premature":
        prediction_date = "2013-09-30"
    elif mutation == "nonbinary":
        development["churn"] = development.churn.astype(float)
        development.loc[0, "churn"] = 0.5
    elif mutation == "missing_feature":
        development = development.drop(columns=FEATURES[0])
    elif mutation == "infinite_feature":
        development.loc[0, "revenue_12m"] = np.inf
    else:
        development = pd.concat([development, development.iloc[[0]]])
    path = tmp_path / "invalid.csv"
    development.to_csv(path, index=False)
    with pytest.raises(ValueError):
        train(path, tmp_path / "model", prediction_date)
    assert not (tmp_path / "model").exists()

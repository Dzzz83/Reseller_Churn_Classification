"""Synthetic model adapters and CSV boundaries; no saved research scores are read."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from reseller_churn.artifacts import (
    file_sha256,
    source_identity,
    write_json,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES, PARAMETERS, SEEDS
from reseller_churn.data.commands import write_preparation
from reseller_churn.modeling.bundles import LoadedEnsemble, load_period_class_model
from reseller_churn.modeling.weighting import period_weights
from reseller_churn.prediction import export_predictions


@pytest.fixture
def retained_model(tmp_path, grouped_data):
    folder = tmp_path / "retained"
    folder.mkdir()
    models = []
    for seed in SEEDS:
        model = HistGradientBoostingClassifier(**PARAMETERS, random_state=seed)
        model.fit(
            grouped_data[list(FEATURES)],
            grouped_data.churn,
            sample_weight=period_weights(grouped_data),
        )
        models.append(model)
    path = folder / "frozen_models.joblib"
    joblib.dump(
        {
            "spec": {
                "id": "period_class_weighted",
                "features": list(FEATURES),
                "parameters": PARAMETERS,
            },
            "models": models,
        },
        path,
    )
    write_json(
        folder / "manifest.json",
        {
            "source": source_identity(),
            "outputs": {
                path.name: file_sha256(path),
                "full_scores.csv": "must-never-be-read",
            },
        },
    )
    return path


def test_period_bundle_adapter_does_not_open_adjacent_scores(retained_model):
    # No full_scores.csv exists, although it is listed in the retained manifest.
    loaded = load_period_class_model(retained_model)
    assert loaded.identity == "period_class_weighted"
    assert loaded.features == FEATURES
    assert len(loaded.models) == 5


@pytest.mark.parametrize(
    "mutation", ["checksum", "runtime", "features", "seed", "spec", "weighting"]
)
def test_invalid_retained_models_are_rejected(retained_model, mutation):
    metadata = retained_model.parent / "manifest.json"
    manifest = json.loads(metadata.read_text())
    if mutation == "checksum":
        retained_model.write_bytes(retained_model.read_bytes() + b"changed")
    elif mutation == "runtime":
        manifest["source"]["versions"]["sklearn"] = "unsupported"
    else:
        bundle = joblib.load(retained_model)
        if mutation == "features":
            bundle["models"][0].feature_names_in_ = np.array(list(reversed(FEATURES)))
        elif mutation == "seed":
            bundle["models"][0].random_state = 999
        elif mutation == "weighting":
            bundle["models"][0].class_weight = "balanced"
        else:
            bundle["spec"]["id"] = "personalized_recency"
        joblib.dump(bundle, retained_model)
        manifest["outputs"][retained_model.name] = file_sha256(retained_model)
    write_json(metadata, manifest)
    with pytest.raises(ValueError):
        load_period_class_model(retained_model)


def test_existing_bundle_export_is_outcome_free_and_does_not_refit(
    tmp_path,
    retained_model,
    source_paths,
    monkeypatch,
):
    folder = tmp_path / "features"
    write_preparation(
        source_paths, "2011-01-01", "2013-10-01", folder, snapshot="2013-10-01"
    )
    frame = pd.read_csv(folder / "features.csv", float_precision="round_trip")
    models = joblib.load(retained_model)["models"]
    expected = np.mean(
        [m.predict_proba(frame[list(FEATURES)])[:, 1] for m in models], axis=0
    )
    monkeypatch.setattr(
        HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("refit")
    )
    output = tmp_path / "export"
    export_predictions(
        retained_model, folder / "features.csv", output, 0.27912545951329487
    )
    result = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    joined = result.set_index("StoreID").reindex(frame.StoreID)
    np.testing.assert_array_equal(joined.churn_probability, expected)
    np.testing.assert_array_equal(
        joined.predicted_churn, expected >= 0.27912545951329487
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["models_fitted"] == 0 and manifest["outcomes_accessed"] is False
    with pytest.raises(FileExistsError):
        export_predictions(
            retained_model, folder / "features.csv", output, 0.27912545951329487
        )


def test_exact_cutoff_equality_survives_feature_and_score_csvs(
    tmp_path, grouped_data, monkeypatch
):
    import reseller_churn.prediction as prediction

    boundary = 0.27912545951329487
    frame = grouped_data.iloc[:2].drop(columns="churn").copy()
    frame["snapshot"] = pd.Timestamp("2014-05-02")
    frame["weighted_discount_3m"] = [boundary, np.nextafter(boundary, 0)]
    folder = tmp_path / "features"
    folder.mkdir()
    frame.to_csv(folder / "features.csv", index=False)
    write_manifest(
        folder,
        {
            "stage": "features",
            "rows": 2,
            "snapshots": ["2014-05-02"],
            "coverage_from": "2013-05-02",
            "coverage_through": "2014-05-02",
            "loaded_through": "2014-05-02",
        },
    )

    class BoundaryModel:
        def predict_proba(self, features):
            score = features.weighted_discount_3m.to_numpy()
            return np.column_stack([1 - score, score])

    model_file = tmp_path / "frozen_models.joblib"
    model_file.write_bytes(b"stub model")
    write_json(tmp_path / "manifest.json", {})
    monkeypatch.setattr(
        prediction,
        "load_period_class_model",
        lambda path: LoadedEnsemble(
            (BoundaryModel(),),
            FEATURES,
            "synthetic",
            "stub",
            "2013-10-01",
        ),
    )
    output = tmp_path / "export"
    export_predictions(model_file, folder / "features.csv", output, boundary)
    result = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    assert result.predicted_churn.tolist() == [1, 0]
    assert result.threshold.eq(boundary).all()


@pytest.mark.parametrize("mutation", ["outcomes", "loaded", "coverage", "checksum"])
def test_export_rejects_untrusted_features(
    tmp_path, retained_model, source_paths, mutation
):
    folder = tmp_path / "features"
    write_preparation(
        source_paths, "2011-01-01", "2013-10-01", folder, snapshot="2013-10-01"
    )
    path = folder / "features.csv"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "outcomes":
        frame = pd.read_csv(path).assign(actual_churn=0)
        frame.to_csv(path, index=False)
        manifest["outputs"][path.name] = file_sha256(path)
    elif mutation == "loaded":
        manifest["loaded_through"] = "2013-09-01"
    elif mutation == "coverage":
        manifest["coverage_from"] = "2013-01-01"
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        export_predictions(
            retained_model, path, tmp_path / "export", 0.27912545951329487
        )
    assert not (tmp_path / "export").exists()

"""Synthetic production contracts; no real data or protected artifacts."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline.smoke_test import source_manifest, synthetic_development
from reseller_churn.artifacts import file_sha256, verify_manifest, write_json
from reseller_churn.config.frozen_model import CUTOFF, FEATURES, SEEDS
from reseller_churn.data.preparation import build_features, load_sources
from reseller_churn.data.source_contract import load_verified_sources
from reseller_churn.modeling.bundles import artifact_manifest, load_model, model_spec
from reseller_churn.modeling.ensemble import apply_threshold
from reseller_churn.modeling.grouped_validation import (
    GroupedValidationPlan,
    ThresholdPolicy,
    cutoff_scan,
)
from reseller_churn.modeling.training import develop, fit_models, validate_development
from reseller_churn.modeling.weighting import period_weights
from reseller_churn.prediction import CSV_COLUMNS, export_predictions, prediction_frame


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-model")
    path = directory / "models.joblib"
    joblib.dump(
        {"spec": model_spec(), "models": fit_models(synthetic_development())}, path
    )
    manifest = directory / "model_manifest.json"
    write_json(manifest, artifact_manifest(path, CUTOFF))
    return path, manifest


@pytest.fixture
def feature_frame(source_paths):
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    return build_features(source, "2013-10-01")


def test_weights_are_partition_local_and_cells_have_equal_totals():
    data = synthetic_development()
    data = data.drop(data[data.StoreID.eq(2)].index[:2]).reset_index(drop=True)
    weights = period_weights(data)
    totals = data.assign(weight=weights).groupby(["snapshot", "churn"]).weight.sum()
    np.testing.assert_allclose(totals, len(data) / 8)
    assert weights.sum() == pytest.approx(len(data))
    with pytest.raises(ValueError, match="Both fitting outcomes"):
        period_weights(data[data.churn.eq(1)])


def test_grouped_partitions_are_disjoint_reproducible_and_complete():
    data = synthetic_development()
    plan = GroupedValidationPlan(outer_folds=3, inner_folds=2, repeat_seeds=(42, 78))
    first, second = list(plan.outer_splits(data)), list(plan.outer_splits(data))
    for (repeat, fold, ti, vi), (_, _, ti2, vi2) in zip(first, second):
        np.testing.assert_array_equal(ti, ti2)
        np.testing.assert_array_equal(vi, vi2)
        assert set(data.iloc[ti].StoreID).isdisjoint(data.iloc[vi].StoreID)
        for fitting, held in plan.inner_splits(data.iloc[ti]):
            assert set(data.iloc[ti].iloc[fitting].StoreID).isdisjoint(
                data.iloc[ti].iloc[held].StoreID
            )
    for repeat in plan.repeat_seeds:
        held = np.concatenate([vi for seed, _, _, vi in first if seed == repeat])
        np.testing.assert_array_equal(np.sort(held), np.arange(len(data)))


def test_cutoff_selector_uses_inclusive_exact_precision_floor():
    scan = cutoff_scan([1, 1, 0, 0, 0], [0.2] * 5)
    chosen = ThresholdPolicy().select(scan)
    assert chosen["cutoff"] == 0.2 and chosen["precision"] == 0.4
    assert ThresholdPolicy(minimum_precision=0.41).select(scan) is None
    np.testing.assert_array_equal(
        apply_threshold([np.nextafter(CUTOFF, 0), CUTOFF], CUTOFF), [0, 1]
    )


def test_bundle_prediction_is_exact_five_seed_mean(model_artifact, feature_frame):
    loaded = load_model(*model_artifact)
    assert len(loaded.models) == 5
    assert [m.random_state for m in loaded.models] == list(SEEDS)
    expected = np.mean(
        [m.predict_proba(feature_frame[list(FEATURES)])[:, 1] for m in loaded.models],
        axis=0,
    )
    np.testing.assert_array_equal(loaded.probabilities(feature_frame), expected)


@pytest.mark.parametrize(
    "mutation",
    ["runtime", "checksum", "features", "spec", "seed", "classes", "order", "count"],
)
def test_bundle_rejects_incompatible_contract(model_artifact, tmp_path, mutation):
    path = tmp_path / "model.joblib"
    path.write_bytes(model_artifact[0].read_bytes())
    metadata = json.loads(model_artifact[1].read_text())
    if mutation in {"runtime", "checksum", "features"}:
        if mutation == "runtime":
            metadata["runtime"]["sklearn"] = "unsupported"
        elif mutation == "checksum":
            metadata["model_sha256"] = "0" * 64
        else:
            metadata["features"] = list(reversed(FEATURES))
    else:
        bundle = joblib.load(path)
        if mutation == "spec":
            bundle["spec"]["id"] = "personalized_recency"
        elif mutation == "seed":
            bundle["models"][0].random_state = 1
        elif mutation == "classes":
            bundle["models"][0].classes_ = np.array([1, 0])
        elif mutation == "order":
            bundle["models"][0].feature_names_in_ = np.array(list(reversed(FEATURES)))
        else:
            bundle["models"] = bundle["models"][:-1]
        joblib.dump(bundle, path)
        metadata["model_sha256"] = file_sha256(path)
    manifest = tmp_path / "model.json"
    write_json(manifest, metadata)
    with pytest.raises(ValueError):
        load_model(path, manifest)


def test_export_exact_features_missingness_precision_and_provenance(
    model_artifact,
    source_paths,
    tmp_path,
    feature_frame,
):
    evidence = source_manifest(source_paths, tmp_path / "coverage.json")
    output = tmp_path / "export"
    kwargs = dict(
        model_path=model_artifact[0],
        model_manifest=model_artifact[1],
        paths=source_paths,
        source_manifest=evidence,
        coverage_from="2011-01-01",
        coverage_through="2013-10-01",
        snapshot="2013-10-01",
        output=output,
    )
    export_predictions(**kwargs)
    csv = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    assert list(csv) == CSV_COLUMNS
    assert csv.StoreID.is_unique and csv.StoreID.tolist() == list(range(2, 9))
    assert set(csv.Snapshot) == {"2013-10-01"}
    pd.testing.assert_frame_equal(
        csv[list(FEATURES)], feature_frame[list(FEATURES)], check_dtype=False
    )
    loaded = load_model(*model_artifact)
    np.testing.assert_array_equal(
        csv["Model score"], loaded.probabilities(feature_frame)
    )
    np.testing.assert_array_equal(
        csv["Predicted churn"].eq("Yes"), csv["Model score"] >= CUTOFF
    )
    raw = (output / "predictions.csv").read_text().splitlines()[1].split(",")
    assert raw[CSV_COLUMNS.index("weighted_discount_3m")] == ""
    manifest = verify_manifest(output)
    assert manifest["rows"] == 7 and manifest["models_fitted"] == 0
    assert manifest["outcomes_accessed"] is False
    assert manifest["threshold"] == CUTOFF
    assert manifest["inputs"]["headers"]["sha256"] == file_sha256(
        source_paths["headers"]
    )
    assert manifest["loaded_through"] == "2013-10-01"
    assert manifest["excluded_order_ids"] == []
    with pytest.raises(FileExistsError):
        export_predictions(**kwargs)


def test_exact_cutoff_is_flagged_before_formatting(feature_frame):
    class BoundaryEnsemble:
        manifest = {"threshold": CUTOFF}

        def probabilities(self, frame):
            assert list(frame[list(FEATURES)]) == list(FEATURES)
            return np.array(
                [CUTOFF, np.nextafter(CUTOFF, 0)] + [0.8] * (len(frame) - 2)
            )

    result = prediction_frame(BoundaryEnsemble(), feature_frame)
    assert result["Predicted churn"].tolist()[:2] == ["Yes", "No"]


@pytest.mark.parametrize(
    "mutation", ["short_details", "hash", "no_attestation", "loaded", "history"]
)
def test_incomplete_effective_coverage_rejected(source_paths, tmp_path, mutation):
    evidence = source_manifest(source_paths, tmp_path / "coverage.json")
    metadata = json.loads(evidence.read_text())
    cutoff = "2013-10-01"
    start = "2011-01-01"
    if mutation == "short_details":
        metadata["sources"]["lines"]["coverage_through"] = "2013-09-01"
    elif mutation == "hash":
        metadata["sources"]["headers"]["sha256"] = "bad"
    elif mutation == "no_attestation":
        metadata.pop("attested_by")
    elif mutation == "loaded":
        cutoff = "2013-11-01"
    else:
        start = "2010-01-01"
    write_json(evidence, metadata)
    with pytest.raises(ValueError):
        load_verified_sources(source_paths, evidence, start, "2013-10-01", cutoff)


def test_duplicate_sources_and_outcomes_fail(source_paths):
    path = source_paths["orders"]
    original = pd.read_csv(path)
    pd.concat([original, original.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    original.assign(churn=0).to_csv(path, index=False)
    with pytest.raises(ValueError, match="outcomes"):
        load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")


@pytest.mark.parametrize(
    "mutation", ["outcomes", "duplicate", "early", "extra_feature"]
)
def test_scoring_boundary_rejects_invalid_rows(model_artifact, feature_frame, mutation):
    loaded = load_model(*model_artifact)
    if mutation == "outcomes":
        feature_frame["churn"] = 0
    elif mutation == "duplicate":
        feature_frame = pd.concat([feature_frame, feature_frame.iloc[[0]]])
    elif mutation == "early":
        feature_frame["snapshot"] = pd.Timestamp("2013-07-01")
    else:
        feature_frame["personalized_recency"] = 1.0
    with pytest.raises(ValueError):
        prediction_frame(loaded, feature_frame)


def test_development_input_rejects_missing_period_and_class():
    data = synthetic_development()
    with pytest.raises(ValueError, match="four approved"):
        validate_development(data[data.snapshot.ne(data.snapshot.min())])
    with pytest.raises(ValueError):
        validate_development(data.assign(churn=2))


def test_no_feasible_cutoff_records_status_without_model(tmp_path, monkeypatch):
    import reseller_churn.modeling.training as training

    monkeypatch.setattr(
        training, "oof_predictions", lambda data, splits: np.full(len(data), 0.5)
    )
    develop(
        synthetic_development(),
        tmp_path / "no-model",
        plan=GroupedValidationPlan(outer_folds=2, inner_folds=2, repeat_seeds=(42,)),
        policy=ThresholdPolicy(minimum_precision=1),
    )
    manifest = verify_manifest(tmp_path / "no-model")
    assert manifest["status"] == "no_feasible_cutoff"
    assert not (tmp_path / "no-model/models.joblib").exists()


def test_proposed_may_snapshot_has_calendar_boundaries(tmp_path):
    from pipeline.smoke_test import write_synthetic_sources

    paths = write_synthetic_sources(
        tmp_path,
        {
            1: ["2013-05-01", "2013-05-02", "2013-11-02", "2014-05-01", "2014-05-02"],
            2: ["2013-11-01"],
            3: ["2013-11-02"],
            4: ["2014-05-02"],
        },
    )
    source = load_sources(paths, "2013-05-02", "2014-05-02", "2014-05-02")
    frame = build_features(source, "2014-05-02")
    assert frame.StoreID.tolist() == [1, 3]
    first = frame[frame.StoreID.eq(1)].iloc[0]
    accepted = source.orders[source.orders.StoreID.eq(1)]
    expected = accepted[accepted.OrderDate.ge("2013-05-02")].SubTotal.sum()
    assert first.revenue_12m == expected
    assert first.recency_days == 1 and first.n_orders_3m == 1
    third = frame[frame.StoreID.eq(3)].iloc[0]
    assert third.n_orders_3m == 0 and np.isnan(third.weighted_discount_3m)
    with pytest.raises(ValueError, match="loaded"):
        build_features(source, "2014-05-03")


@pytest.mark.parametrize(
    "mutation",
    ["intraday", "negative_categories", "fractional_quantity", "duplicate_columns"],
)
def test_strict_source_values_fail(source_paths, mutation):
    path = source_paths["orders"]
    frame = pd.read_csv(path)
    if mutation == "intraday":
        frame.loc[0, "OrderDate"] = "2012-03-01T12:00:00"
    elif mutation == "negative_categories":
        frame.loc[0, "rev_bikes"] = -1
        frame.loc[0, "rev_components"] = frame.loc[0, "SubTotal"] + 1
    elif mutation == "fractional_quantity":
        frame.loc[0, "qty"] = 1.5
    else:
        original = path.read_text().splitlines()
        original[0] += ",StoreID"
        original[1:] = [line + ",1" for line in original[1:]]
        path.write_text("\n".join(original) + "\n")
    if mutation != "duplicate_columns":
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")


def test_grouped_metrics_report_false_alerts_and_incomplete_repeats():
    from reseller_churn.evaluation import assessment_metrics

    rows = pd.DataFrame(
        {
            "repeat": [42] * 4,
            "snapshot": ["synthetic"] * 4,
            "churn": [1, 1, 0, 0],
            "predicted_churn": [True, False, True, False],
        }
    )
    overall = assessment_metrics(rows, 5).iloc[0]
    assert overall.recall == 0.5 and overall.precision == 0.5
    assert overall.false_alerts == 1 and not overall.complete_repeat


@pytest.mark.parametrize(
    "threshold", [None, True, float("nan"), float("inf"), -0.1, 1.1]
)
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        apply_threshold([0.5], threshold)


def test_retained_identity_cannot_override_cutoff(
    model_artifact, tmp_path, monkeypatch
):
    import reseller_churn.modeling.bundles as bundles

    metadata = json.loads(model_artifact[1].read_text())
    from reseller_churn.config.frozen_model import RETAINED_FEATURES

    monkeypatch.setattr(bundles, "RETAINED_SHA256", metadata["model_sha256"])
    metadata["features"] = list(RETAINED_FEATURES)
    metadata["threshold"] = 0.27912546
    path = tmp_path / "modified.json"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="exact frozen cutoff"):
        load_model(model_artifact[0], path)


def test_retained_feature_name_adapter_preserves_scores_and_public_csv(
    model_artifact, feature_frame, source_paths, tmp_path, monkeypatch
):
    import reseller_churn.modeling.bundles as bundles
    from reseller_churn.config.frozen_model import RETAINED_FEATURES

    # Emulate the retained artifact's original names on a synthetic fitted
    # ensemble. Its numeric state and feature positions remain unchanged.
    bundle = joblib.load(model_artifact[0])
    bundle["spec"]["features"] = list(RETAINED_FEATURES)
    for model in bundle["models"]:
        model.feature_names_in_ = np.array(RETAINED_FEATURES)
    path = tmp_path / "retained.joblib"
    joblib.dump(bundle, path)
    metadata = json.loads(model_artifact[1].read_text())
    metadata["features"] = list(RETAINED_FEATURES)
    metadata["model_sha256"] = file_sha256(path)
    manifest_path = tmp_path / "retained.json"
    write_json(manifest_path, metadata)
    monkeypatch.setattr(bundles, "RETAINED_SHA256", metadata["model_sha256"])
    expected_inputs = feature_frame[list(FEATURES)].copy()
    expected_inputs.columns = RETAINED_FEATURES
    expected = np.mean(
        [model.predict_proba(expected_inputs)[:, 1] for model in bundle["models"]],
        axis=0,
    )
    np.testing.assert_array_equal(
        load_model(path, manifest_path).probabilities(feature_frame), expected
    )
    evidence = source_manifest(source_paths, tmp_path / "source_coverage.json")
    output = tmp_path / "export"
    export_predictions(
        model_path=path,
        model_manifest=manifest_path,
        paths=source_paths,
        source_manifest=evidence,
        coverage_from="2011-01-01",
        coverage_through="2013-10-01",
        snapshot="2013-10-01",
        output=output,
    )
    csv = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    assert "discount_frequency_12m" in csv
    assert "discounted_line_share_12m" not in csv
    np.testing.assert_array_equal(csv["Model score"], expected)
    np.testing.assert_array_equal(
        csv.discount_frequency_12m, feature_frame.discount_frequency_12m
    )
    saved = verify_manifest(output)
    assert saved["features"] == list(FEATURES)
    assert saved["estimator_features"] == list(RETAINED_FEATURES)
    assert saved["feature_name_mapping"] == {
        "discount_frequency_12m": "discounted_line_share_12m"
    }

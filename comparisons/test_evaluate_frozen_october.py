"""Synthetic-only checks for frozen scoring and explicit outcome evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from comparisons import evaluate_frozen_october as final
from comparisons import grouped_precision_recall as study


def feature_frame(rows=2):
    frame = pd.DataFrame(
        {name: np.zeros(rows) for name in study.FEATURES + study.CADENCE}
    )
    frame["StoreID"] = np.arange(1, rows + 1)
    frame["snapshot"] = final.SNAPSHOT
    frame["score"] = 0.1
    frame["n_orders_3m"] = 1
    return frame


def test_labeled_scoring_input_is_rejected():
    frame = feature_frame().assign(churn=0)
    with pytest.raises(ValueError, match="unlabeled"):
        final.validate_features(frame, study.FEATURES + study.CADENCE, expected_rows=2)


def test_wrong_snapshot_duplicate_and_infinite_features_rejected():
    for mutation in ("snapshot", "duplicate", "infinite"):
        frame = feature_frame()
        if mutation == "snapshot":
            frame["snapshot"] = "2014-01-01"
        if mutation == "duplicate":
            frame.loc[1, "StoreID"] = 1
        if mutation == "infinite":
            frame.loc[0, "gap_cv"] = np.inf
        with pytest.raises(ValueError):
            final.validate_features(
                frame, study.FEATURES + study.CADENCE, expected_rows=2
            )


def test_supplied_outcomes_must_match_scored_population_exactly():
    predictions = feature_frame()[study.KEYS].assign(strategy="synthetic")
    labels = predictions[study.KEYS].assign(churn=[1, 0])
    assert final.matched_outcomes(
        predictions, labels, expected_rows=2
    ).churn.tolist() == [1, 0]
    with pytest.raises(ValueError, match="populations differ"):
        final.matched_outcomes(
            predictions, labels.assign(StoreID=[1, 3]), expected_rows=2
        )
    with pytest.raises(ValueError, match="unique"):
        final.matched_outcomes(
            predictions, pd.concat([labels, labels.iloc[[0]]]), expected_rows=3
        )


def test_explicit_approval_is_checked_before_reading_any_inputs(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        final.evaluate(
            tmp_path / "missing", tmp_path / "missing.csv", tmp_path / "missing.json"
        )


def test_changed_checkpoint_is_rejected_before_outcome_access(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "prediction_checkpoint.json").write_text(
        json.dumps({"protocol_sha256": "changed", "predictions_sha256": "changed"})
    )
    with pytest.raises(ValueError, match="checkpoint changed"):
        final.evaluate(
            tmp_path,
            tmp_path / "missing.csv",
            tmp_path / "missing.json",
            approve_inspected=True,
        )


def test_artifact_hash_tampering_is_rejected(tmp_path):
    path = tmp_path / "artifact.csv"
    path.write_text("original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"outputs": {path.name: study.file_hash(path)}}))
    final.verified_artifact(manifest, path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        final.verified_artifact(manifest, path)


def test_synthetic_two_stage_run_never_fits_and_keeps_frozen_cutoff(
    tmp_path, monkeypatch
):
    frame = feature_frame(492)
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    checkpoint = tmp_path / "prediction_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            dict(
                score_sha256=study.file_hash(path),
                cutoff=0.06,
                features=study.FEATURES,
                model=study.BASE_PARAMETERS,
                seeds=study.MODEL_SEEDS,
            )
        )
    )
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    path.name: study.file_hash(path),
                    checkpoint.name: study.file_hash(checkpoint),
                }
            }
        )
    )

    class Estimator:
        def fit(self, *args, **kwargs):
            raise AssertionError("Evaluation must not train")

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    spec = dict(candidate=dict(features=study.FEATURES + study.CADENCE), cutoff=0.5)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "research_candidate.joblib").write_bytes(b"synthetic-not-loaded")
    (model_directory / "manifest.json").write_text("{}")
    monkeypatch.setattr(final, "load_frozen_model", lambda _: ([Estimator()], spec, {}))
    output = tmp_path / "output"
    final.predict(model_directory, path, manifest, output)
    saved = json.loads((output / "prediction_checkpoint.json").read_text())
    assert saved["outcomes_read_by_this_stage"] is False
    labels = frame[study.KEYS].assign(churn=(frame.StoreID <= 46).astype(int))
    outcome_path = tmp_path / "outcomes.csv"
    labels.to_csv(outcome_path, index=False)
    outcome_manifest = tmp_path / "outcomes.json"
    outcome_manifest.write_text(
        json.dumps({"outputs": {outcome_path.name: study.file_hash(outcome_path)}})
    )
    final.evaluate(output, outcome_path, outcome_manifest, approve_inspected=True)
    metrics = study.read_csv(output / "metrics.csv")
    selected = metrics[
        metrics.strategy.eq("frozen_cadence_candidate") & metrics.subgroup.eq("overall")
    ].iloc[0]
    assert selected.tp == 46 and selected.fp == 446
    assert selected.cutoff == 0.5
    with pytest.raises(FileExistsError):
        final.evaluate(output, outcome_path, outcome_manifest, approve_inspected=True)

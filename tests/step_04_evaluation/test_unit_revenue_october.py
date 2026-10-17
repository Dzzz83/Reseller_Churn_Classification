"""Focused checks for the authorized latest-candidate October evaluation."""

import numpy as np
import pandas as pd
import pytest

from comparisons import evaluate_unit_revenue_october as final
from reseller_churn.artifacts import write_json, write_manifest


def test_reconciliation_aligns_keys_and_preserves_missingness():
    original = pd.DataFrame(
        dict(StoreID=[1, 2], snapshot=["2013-10-01"] * 2, feature=[np.nan, 2.0])
    )
    final.reconcile(original, original.iloc[::-1], ["feature"])
    with pytest.raises(AssertionError):
        final.reconcile(original, original.fillna(0), ["feature"])
    with pytest.raises(ValueError, match="populations"):
        final.reconcile(original, original.assign(StoreID=[1, 3]), ["feature"])


def test_paired_change_signs_and_resampling_match_operational_goal():
    baseline = pd.DataFrame(
        dict(
            StoreID=[1, 2, 3, 4],
            snapshot=["2013-10-01"] * 4,
            churn=[1, 1, 0, 0],
            predicted_churn=[1, 0, 1, 1],
            strategy="baseline",
        )
    )
    candidate = baseline.assign(strategy=final.CANDIDATE, predicted_churn=[1, 1, 0, 1])
    cases = pd.concat([baseline, candidate], ignore_index=True)
    summary = final.paired_summary(cases, "baseline", draws=100)
    assert summary["newly_caught"] == 1
    assert summary["newly_missed"] == 0
    assert summary["false_alerts_removed"] == 1
    assert summary["false_alerts_added"] == 0
    assert summary["recall_difference"] == 0.5
    assert summary["false_alert_reduction"] == 1
    assert summary["point_objective_met"]
    assert summary == final.paired_summary(cases, "baseline", draws=100)
    candidate.predicted_churn = [0, 0, 0, 0]
    assert not final.paired_summary(
        pd.concat([baseline, candidate]), "baseline", draws=100
    )["point_objective_met"]


def test_checkpoint_tamper_rejected_before_any_outcome_access(tmp_path):
    write_json(tmp_path / "protocol.json", {"inputs": {}})
    write_manifest(
        tmp_path, dict(stage="predictions_checkpointed", outcomes_read=False)
    )
    final.checked_prediction_stage(tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        final.evaluate(tmp_path)
    assert not (tmp_path / "evaluation").exists()


def test_changed_external_input_rejected(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("original")
    output = tmp_path / "output"
    output.mkdir()
    write_json(
        output / "protocol.json",
        {
            "inputs": {
                "source": {"path": str(source), "sha256": final.study.file_hash(source)}
            }
        },
    )
    write_manifest(output, dict(stage="predictions_checkpointed", outcomes_read=False))
    source.write_text("changed")
    with pytest.raises(ValueError, match="input or code changed"):
        final.checked_prediction_stage(output)


def test_evaluation_completes_report_without_optional_markdown_package(
    tmp_path, monkeypatch
):
    output = tmp_path / "predictions"
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    keys = pd.DataFrame({"StoreID": range(1, 493), "snapshot": ["2013-10-01"] * 492})
    labels = keys.assign(churn=[1, 1] + [0] * 490)
    labels.to_csv(source / "october_cases.csv", index=False)
    write_manifest(source, {})
    parts = []
    for name in ["baseline", "historical_0.06", final.CANDIDATE]:
        parts.append(
            keys.assign(
                strategy=name,
                score=0.5,
                cutoff=0.2,
                n_orders_3m=[0, 1] * 246,
                predicted_churn=1,
            )
        )
    pd.concat(parts).to_csv(output / "unlabeled_predictions.csv", index=False)
    write_json(
        output / "protocol.json",
        {"inputs": {}, "bootstrap_draws": 50, "bootstrap_seed": 2026},
    )
    write_manifest(output, dict(stage="predictions_checkpointed", outcomes_read=False))
    monkeypatch.setattr(final, "SAVED", source)
    final.evaluate(output)
    assert (output / "evaluation/report.md").exists()
    assert (output / "evaluation/manifest.json").exists()
    with pytest.raises(FileExistsError):
        final.evaluate(output)

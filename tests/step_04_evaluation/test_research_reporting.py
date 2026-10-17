"""Confirmation reporting preserves the frozen primary decision."""

import json

import pytest

from comparisons.breadth_reporting import load_decision
from reseller_churn.artifacts import file_sha256


def test_confirmation_cannot_rescue_failed_primary(tmp_path):
    primary = tmp_path / "decision.json"
    primary.write_text(json.dumps({"accepted": False}))
    final = tmp_path / "final_decision.json"
    final.write_text(
        json.dumps(
            {
                "primary_decision_sha256": file_sha256(primary),
                "decision": {"accepted": True},
            }
        )
    )
    with pytest.raises(ValueError, match="cannot rescue"):
        load_decision(tmp_path)


def test_confirmation_binds_primary_and_can_reject(tmp_path):
    primary = tmp_path / "decision.json"
    primary.write_text(json.dumps({"accepted": True}))
    final = tmp_path / "final_decision.json"
    final.write_text(
        json.dumps(
            {
                "primary_decision_sha256": file_sha256(primary),
                "decision": {"accepted": False, "bootstrap_draws": 50000},
            }
        )
    )
    assert load_decision(tmp_path) == {"accepted": False, "bootstrap_draws": 50000}
    primary.write_text(json.dumps({"accepted": False}))
    with pytest.raises(ValueError, match="different primary"):
        load_decision(tmp_path)

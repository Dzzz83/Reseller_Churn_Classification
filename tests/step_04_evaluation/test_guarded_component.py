"""Guarded OOF selection preserves the baseline and never silently drops a fold."""

import json

import numpy as np
import pandas as pd

from comparisons.additional_uncertainty import additional_confirmation
from comparisons.guarded_component import COMPONENT, choose_component, guarded_scan


def reference_frame():
    frame = pd.DataFrame(
        {
            "StoreID": range(1, 41),
            "snapshot": pd.Timestamp("2012-07-01"),
            "churn": [1] * 20 + [0] * 20,
            "n_orders_3m": np.tile([1, 0], 20),
            "score": [0.9] * 18 + [0.4] * 2 + [0.6] * 10 + [0.2] * 10,
        }
    )
    return frame


def test_common_objective_baseline_parity_and_successful_training_component():
    reference = reference_frame()
    candidate = reference.assign(score=[0.9] * 20 + [0.95] * 2 + [0.1] * 18)
    choice, scan, _ = choose_component(reference, candidate)
    assert choice["reference_cutoff"] == 0.4
    assert choice["component"] == COMPONENT and choice["cutoff"] == 0.9
    row = scan[scan.component.eq(COMPONENT) & scan.cutoff.eq(0.9)].iloc[0]
    assert row.tp == 20 and row.fp == 2 and row.feasible
    # Equality alerts the entire tied positive block.
    assert candidate.score.ge(choice["cutoff"]).sum() == 22


def test_same_scores_fall_back_when_practical_gain_is_absent():
    reference = reference_frame()
    choice, _, _ = choose_component(reference, reference)
    assert choice["component"] == "baseline" and choice["cutoff"] == 0.4
    assert choice["reason"] == "inner_practical_gain_not_met"


def test_baseline_recall_parity_and_group_fp_guards_can_make_component_infeasible():
    reference = reference_frame()
    candidate = reference.assign(score=[0.9] * 19 + [0.05] + [0.95] * 2 + [0.1] * 18)
    choice, _, _ = choose_component(reference, candidate)
    assert choice["component"] == "baseline"
    assert choice["reason"] == "no_feasible_component_cutoff"
    assert choice["component_feasible_cutoffs"] == 0


def test_exact_five_point_group_recall_boundary_uses_counts():
    reference = pd.DataFrame(
        {
            "StoreID": range(1, 81),
            "snapshot": pd.Timestamp("2012-07-01"),
            "churn": [1] * 40 + [0] * 40,
            "n_orders_3m": np.tile([1, 0], 40),
            "score": [0.9] * 40 + [0.6] * 20 + [0.2] * 20,
        }
    )
    candidate = reference.assign(score=[0.1] + [0.9] * 39 + [0.6] * 10 + [0.2] * 30)
    scan, _ = guarded_scan(reference, candidate, 0.5)
    point = scan[scan.cutoff.eq(0.6)].iloc[0]
    assert point.group_recall_passed
    assert not point.pooled_recall_passed


def test_undefined_training_group_is_explicit_baseline_fallback():
    reference = reference_frame().assign(n_orders_3m=1)
    choice, _, groups = choose_component(
        reference, reference.assign(score=[0.9] * 20 + [0.1] * 20)
    )
    assert (
        choice["component"] == "baseline"
        and choice["reason"] == "undefined_inner_group"
    )
    assert not groups.recall_defined.all()


def test_failed_primary_never_receives_more_bootstrap_draws(tmp_path, monkeypatch):
    primary = {"accepted": False, "checks": {"recall_preserved": False}}
    (tmp_path / "decision.json").write_text(json.dumps(primary))

    def forbidden(*args, **kwargs):
        raise AssertionError("Failed primary must not be rescued")

    monkeypatch.setattr(
        "comparisons.additional_uncertainty.paired_bootstrap", forbidden
    )
    monkeypatch.setattr(
        "comparisons.additional_uncertainty.simulation_audit", forbidden
    )
    final = additional_confirmation(tmp_path, "unused", 0.1)
    assert final == primary
    record = json.loads((tmp_path / "final_decision.json").read_text())
    assert set(record["stages"].values()) == {"not_run_primary_failed"}
    assert not (tmp_path / "confirmation").exists()


def test_binding_intervals_distinguish_feasible_and_conflicting_training_guards():
    from comparisons.guard_binding_diagnosis import bounds_for_partition

    reference = reference_frame()
    eligible = reference.assign(score=[0.9] * 20 + [0.95] * 2 + [0.1] * 18)
    _, feasible = bounds_for_partition(reference, eligible, 0.4)
    assert (
        feasible["feasible"]
        and feasible["fp_lower_index"] <= feasible["recall_upper_index"]
    )
    conflicting = reference.assign(score=[0.9] * 19 + [0.05] + [0.95] * 2 + [0.1] * 18)
    _, failure = bounds_for_partition(reference, conflicting, 0.4)
    assert not failure["feasible"]
    assert failure["fp_required_cutoff"] > failure["recall_max_cutoff"]

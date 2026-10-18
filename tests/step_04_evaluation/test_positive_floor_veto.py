"""Training floor ties, cutoff parity and outer alert-subset behavior."""

import json

import numpy as np
import pytest

from comparisons.positive_floor_veto import apply_veto, exact_confirmation, learn_veto
from tests.step_04_evaluation.test_guarded_component import reference_frame


def test_inner_floor_retains_ties_and_every_baseline_catch():
    baseline = reference_frame()
    component = baseline.assign(score=[0.5] * 20 + [0.5] * 2 + [0.1] * 18)
    selection, transformed, protected = learn_veto(baseline, component)
    assert selection["reference_cutoff"] == 0.4
    assert selection["veto_floor"] == 0.5
    assert len(protected) == 20
    assert transformed.score.ge(0.4).sum() == 22
    assert transformed.loc[:19, "score"].ge(0.4).all()


def test_outer_floor_can_lose_catches_but_never_add_alerts():
    baseline = reference_frame()
    component = baseline.assign(score=[0.1] + [0.5] * 19 + [0.1] * 10 + [0.9] * 10)
    selected = dict(reference_cutoff=0.4, veto_floor=0.5, fallback=False)
    result = apply_veto(baseline, component, selected)
    assert not result.iloc[0].score >= 0.4
    assert np.all(~result.score.ge(0.4) | baseline.score.ge(0.4))


def test_zero_cutoff_falls_back_and_case_mismatch_fails():
    baseline = reference_frame().assign(score=0.0)
    component = baseline.assign(score=0.1)
    selected, result, _ = learn_veto(baseline, component)
    assert selected["fallback"] and result.score.eq(0.0).all()
    with pytest.raises(AssertionError):
        apply_veto(
            baseline, component.assign(StoreID=lambda x: x.StoreID + 1), selected
        )


def test_failed_primary_does_not_run_confirmation(tmp_path, monkeypatch):
    primary = dict(accepted=False, checks=dict(recall=False))
    (tmp_path / "decision.json").write_text(json.dumps(primary))

    def forbidden(*args, **kwargs):
        raise AssertionError("A failed primary cannot receive rescue draws")

    monkeypatch.setattr("comparisons.positive_floor_veto.paired_bootstrap", forbidden)
    monkeypatch.setattr("comparisons.positive_floor_veto.sign_audit", forbidden)
    assert exact_confirmation(tmp_path) == primary

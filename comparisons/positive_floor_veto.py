"""Training-positive floor veto over unchanged baseline alerts; no new fits."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.development_reporting import cutoff_transfer_tables
from comparisons.exact_bootstrap_signs import sign_audit
from comparisons.guarded_component import count_curve
from comparisons.matched_evaluation import (
    acceptance,
    paired_bootstrap,
    publish_assessment,
)
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
CANDIDATE = "positive_floor_veto"
COMPONENT = "retained_logistic"
TAIL = 0.00000048828125


def align(reference, component):
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    component = component.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        reference[KEYS + ["churn"]], component[KEYS + ["churn"]]
    )
    require(not reference.duplicated(KEYS).any(), "Duplicate score cases")
    for rows in [reference, component]:
        require(
            np.isfinite(rows.score).all() and rows.score.between(0, 1).all(),
            "Invalid scores",
        )
    return reference, component


def constrained_cutoff(reference, candidate, cutoff):
    """One common training objective, with the original catches as its TP floor."""
    cuts = np.unique(np.r_[cutoff, candidate.score])
    tp, fp = count_curve(candidate, cuts)
    target = int((reference.churn.eq(1) & reference.score.ge(cutoff)).sum())
    feasible = (cuts >= cutoff) & (tp >= target) & (tp / reference.churn.sum() > 0.90)
    require(feasible.any(), "No feasible training cutoff")
    precision = np.divide(
        tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0
    )
    options = np.flatnonzero(feasible)
    best = max(options, key=lambda i: (precision[i], tp[i], cuts[i]))
    return float(cuts[best])


def apply_veto(reference, component, selection):
    """Apply only the saved training floor; equality keeps the whole score tie."""
    reference, component = align(reference, component)
    result = reference.copy()
    if not selection["fallback"]:
        result["score"] = reference.score.where(
            component.score.ge(selection["veto_floor"]), 0.0
        )
    cutoff = selection["reference_cutoff"]
    require(
        (~result.score.ge(cutoff) | reference.score.ge(cutoff)).all(),
        "Veto added an alert",
    )
    return result


def learn_veto(reference, component):
    """Accept inner OOF data only. No outer scores enter floor estimation."""
    reference, component = align(reference, component)
    cutoff = float(study.select_threshold(reference.churn, reference.score)["cutoff"])
    caught = reference.churn.eq(1) & reference.score.ge(cutoff)
    require(caught.any(), "No caught inner churners")
    selection = dict(
        reference_cutoff=cutoff,
        veto_floor=float(component.loc[caught, "score"].min()),
        protected_inner_cases=int(caught.sum()),
        fallback=cutoff == 0.0,
    )
    transformed = apply_veto(reference, component, selection)
    for candidate in [reference, transformed]:
        require(
            constrained_cutoff(reference, candidate, cutoff) == cutoff,
            "Common cutoff changed",
        )
    require(
        transformed.loc[caught, "score"].ge(cutoff).all(),
        "Lost a protected inner catch",
    )
    protected = reference.loc[caught, KEYS + ["churn", "score"]].copy()
    protected["component_score"] = component.loc[caught, "score"].to_numpy()
    return selection, transformed, protected


def exact_confirmation(output):
    """Additional frozen gates only; a failed primary is never rescued."""
    primary = json.loads((output / "decision.json").read_text())
    final = copy.deepcopy(primary)
    stages = dict(
        confirmation="not_run_primary_failed", exact_signs="not_run_primary_failed"
    )
    if primary["accepted"]:
        predictions = read_csv(output / "validation_predictions.csv")
        metrics = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
        folder = new_output_directory(output / "confirmation")
        draws = paired_bootstrap(
            predictions, draws=50000, seed=2026, candidate_id=CANDIDATE
        )
        draws.to_csv(folder / "bootstrap.csv", index=False)
        decision, _, _ = acceptance(metrics, draws, TAIL, CANDIDATE)
        write_json(folder / "decision.json", decision)
        final["checks"]["fifty_thousand_confirmation"] = decision["accepted"]
        stages["confirmation"] = "passed" if decision["accepted"] else "failed"
        stages["exact_signs"] = "not_run_confirmation_failed"
        if decision["accepted"]:
            audit = sign_audit(predictions, CANDIDATE, TAIL)
            write_json(folder / "exact_sign_audit.json", audit)
            final["checks"]["exact_empirical_signs"] = audit[
                "conditional_sign_gates_passed"
            ]
            stages["exact_signs"] = (
                "passed" if audit["conditional_sign_gates_passed"] else "failed"
            )
        write_manifest(
            folder, dict(stage="prospective_exact_confirmation", models_fitted=0)
        )
    final["accepted"] = all(final["checks"].values())
    write_json(
        output / "final_decision.json",
        dict(
            primary_decision_sha256=file_sha256(output / "decision.json"),
            decision=final,
            stages=stages,
        ),
    )
    return final


def run(output):
    source = ROOT / "results/retained-calibration-20261018"
    verify_manifest(source)
    data = read_csv(source / "development_input.csv")
    previous = json.loads((source / "protocol.json").read_text())
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_source)
    history = read_csv(history_source / "case_histories.csv")
    profiles = history[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol_path = ROOT / "comparisons/positive_floor_veto_protocol.md"
    shutil.copyfile(protocol_path, out / "protocol.md")
    snapshot = new_output_directory(out / "source_snapshot")
    implementations = [
        Path(__file__),
        ROOT / "comparisons/exact_bootstrap_signs.py",
        ROOT / "comparisons/matched_evaluation.py",
        ROOT / "comparisons/guarded_component.py",
        ROOT / "comparisons/retained_calibration.py",
    ]
    for path in implementations:
        shutil.copyfile(path, snapshot / path.name)
    data.to_csv(out / "development_input.csv", index=False)
    spec = dict(
        id=CANDIDATE,
        description="Baseline alert veto below minimum caught-inner-churner calibrated score",
        features=previous["arms"][1]["features"],
        components=[dict(id="crossfit_components")],
    )
    protocol = dict(
        arms=[previous["arms"][0], spec],
        splits=previous["splits"],
        repeat_seeds=previous["repeat_seeds"],
        period_limits=LIMITS,
        interval_tail_probability=TAIL,
        alpha_spending_cycle=10,
        new_fits=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "preexecution_identity.json",
        dict(
            inputs=input_identity(
                {"source": source / "manifest.json", "protocol": protocol_path}
            ),
            implementation=input_identity({p.stem: p for p in implementations}),
        ),
    )
    choices, parts, transfers = [], [], []
    for split in previous["splits"]:
        train, _ = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        original = source / relative
        verify_manifest(original)
        folder = new_output_directory(out / relative)
        inner = read_csv(original / "inner_scores.csv")
        baseline = inner[inner.candidate.eq("baseline")]
        alternative = inner[inner.candidate.eq(COMPONENT)]
        selected, transformed, protected = learn_veto(baseline, alternative)
        write_json(folder / "selection.json", selected)
        protected.to_csv(folder / "protected_inner_cases.csv", index=False)
        # The floor is saved before outer predictions or labels are opened.
        outer = read_csv(original / "predictions.csv")
        original_baseline = outer[outer.strategy.eq("baseline")]
        require(
            original_baseline.cutoff.eq(selected["reference_cutoff"]).all(),
            "Baseline cutoff changed",
        )
        chosen = apply_veto(
            original_baseline, outer[outer.strategy.eq(COMPONENT)], selected
        ).assign(strategy=CANDIDATE)
        predictions = pd.concat([original_baseline, chosen], ignore_index=True)
        saved_inner = pd.concat(
            [baseline, transformed.assign(candidate=CANDIDATE)], ignore_index=True
        )
        saved_inner.to_csv(folder / "inner_scores.csv", index=False)
        predictions.to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(
            [
                dict(candidate=c, cutoff=selected["reference_cutoff"])
                for c in ["baseline", CANDIDATE]
            ]
        ).to_csv(folder / "inner_selections.csv", index=False)
        for name in ["model_identities.json", "cached_hgb_identities.json"]:
            shutil.copyfile(original / name, folder / name)
        write_manifest(
            folder,
            dict(
                new_fits=0,
                inputs=input_identity({"source": original / "manifest.json"}),
            ),
        )
        choices.append(
            dict(repeat=split["repeat"], outer_fold=split["outer_fold"], **selected)
        )
        parts.append(predictions)
        transfers.append(cutoff_transfer_tables(saved_inner, predictions, data))
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "floor_choices.csv", index=False)
    pd.concat(transfers, ignore_index=True).to_csv(
        out / "cutoff_transfer.csv", index=False
    )
    child = out / CANDIDATE
    publish_assessment(child, out, data, profiles, predictions, protocol)
    primary = intersection_gate(child)
    write_json(
        child / "completion_manifest.json",
        dict(outputs={p.name: file_sha256(p) for p in child.iterdir() if p.is_file()}),
    )
    final = exact_confirmation(child)
    write_json(
        out / "final_procedure_decision.json", dict(primary=primary, final=final)
    )
    write_manifest(
        out,
        dict(
            stage=CANDIDATE,
            fits=0,
            accepted=final["accepted"],
            inputs=input_identity({"source": source / "manifest.json"}),
        ),
    )
    print(pd.DataFrame(choices).to_string(index=False))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

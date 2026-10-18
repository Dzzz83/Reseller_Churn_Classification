"""One training-only guarded component choice with an explicit original-baseline fallback."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.additional_uncertainty import additional_confirmation
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.development_checks import validate_development_cases
from comparisons.development_reporting import cutoff_transfer_tables, slices
from comparisons.matched_evaluation import publish_assessment
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
CANDIDATE = "guarded_component"
COMPONENT = "retained_logistic"
TAIL = 0.0000009765625


def count_curve(frame, cutoffs):
    """Enumerate complete tied-score blocks; equality always alerts."""
    positive = np.sort(frame.loc[frame.churn.eq(1), "score"].to_numpy())
    negative = np.sort(frame.loc[frame.churn.eq(0), "score"].to_numpy())
    tp = len(positive) - np.searchsorted(positive, cutoffs, side="left")
    fp = len(negative) - np.searchsorted(negative, cutoffs, side="left")
    return tp, fp


def guarded_scan(reference, candidate, reference_cutoff):
    """Apply the same baseline-relative inner constraints to either score vector."""
    fields = KEYS + ["churn", "n_orders_3m"]
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    candidate = candidate.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(reference[fields], candidate[fields])
    validate_development_cases(reference)
    for rows in [reference, candidate]:
        require(
            np.isfinite(rows.score).all() and rows.score.between(0, 1).all(),
            "Invalid inner scores",
        )
    cutoffs = np.unique(np.r_[0.0, candidate.score])
    tp, fp = count_curve(candidate, cutoffs)
    base_alert = reference.score.ge(reference_cutoff)
    base_tp = int((base_alert & reference.churn.eq(1)).sum())
    curve = pd.DataFrame(
        dict(
            cutoff=cutoffs,
            tp=tp,
            fp=fp,
            recall=tp / reference.churn.sum(),
            precision=np.divide(
                tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) > 0
            ),
        )
    )
    curve["pooled_recall_passed"] = curve.recall.gt(0.90) & curve.tp.ge(base_tp)
    curve["group_recall_passed"] = True
    curve["group_fp_passed"] = True
    curve["worst_fp_increase"] = -np.inf
    curve["worst_recall_loss"] = -np.inf
    groups, undefined = [], 0
    for dimension, name, part in slices(reference):
        if dimension == "overall":
            continue
        group_tp, group_fp = count_curve(candidate.loc[part.index], cutoffs)
        positives = int(part.churn.sum())
        alert = part.score.ge(reference_cutoff)
        target_tp = int((alert & part.churn.eq(1)).sum())
        target_fp = int((alert & part.churn.eq(0)).sum())
        groups.append(
            dict(
                slice=dimension,
                group=name,
                rows=len(part),
                churners=positives,
                baseline_tp=target_tp,
                baseline_fp=target_fp,
                recall_defined=positives > 0,
            )
        )
        if positives == 0:
            curve["group_recall_passed"] = False
            undefined += 1
        else:
            curve["group_recall_passed"] &= (group_tp - target_tp) >= -0.05 * positives
            curve["worst_recall_loss"] = np.maximum(
                curve.worst_recall_loss, (target_tp - group_tp) / positives
            )
        curve["group_fp_passed"] &= group_fp <= target_fp
        curve["worst_fp_increase"] = np.maximum(
            curve.worst_fp_increase, group_fp - target_fp
        )
    curve["feasible"] = (
        curve.pooled_recall_passed & curve.group_recall_passed & curve.group_fp_passed
    )
    curve["undefined_reference_groups"] = undefined
    return curve, pd.DataFrame(groups)


def best_cutoff(curve):
    feasible = curve[curve.feasible]
    if feasible.empty:
        return None
    return (
        feasible.sort_values(["precision", "recall", "cutoff"], ascending=False)
        .iloc[0]
        .to_dict()
    )


def choose_component(reference, candidate):
    """Only training OOF labels are accepted; no outer scores or labels enter."""
    cutoff = study.select_threshold(reference.churn, reference.score)["cutoff"]
    baseline_scan, groups = guarded_scan(reference, reference, cutoff)
    candidate_scan, _ = guarded_scan(reference, candidate, cutoff)
    baseline_choice = best_cutoff(baseline_scan)
    if not groups.recall_defined.all():
        reason, choice = "undefined_inner_group", None
    else:
        require(
            baseline_choice is not None and baseline_choice["cutoff"] == cutoff,
            "Common constrained baseline differs from original reference",
        )
        choice = best_cutoff(candidate_scan)
        reason = (
            "no_feasible_component_cutoff"
            if choice is None
            else "inner_practical_gain_not_met"
        )
    reference_fp = int((reference.churn.eq(0) & reference.score.ge(cutoff)).sum())
    use_component = (
        choice is not None
        and choice["fp"] < reference_fp
        and (reference_fp - choice["fp"]) * 20 >= reference_fp
    )
    if use_component:
        reason = "inner_guards_and_practical_gain_passed"
    selected = dict(
        component=COMPONENT if use_component else "baseline",
        cutoff=float(choice["cutoff"] if use_component else cutoff),
        reason=reason,
        reference_cutoff=float(cutoff),
        reference_fp=reference_fp,
        component_feasible_cutoffs=int(candidate_scan.feasible.sum()),
        component_cutoff=None if choice is None else float(choice["cutoff"]),
        component_fp=None if choice is None else int(choice["fp"]),
    )
    scans = pd.concat(
        [
            baseline_scan.assign(component="baseline"),
            candidate_scan.assign(component=COMPONENT),
        ],
        ignore_index=True,
    )
    return selected, scans, groups


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
    protocol_path = ROOT / "comparisons/guarded_component_protocol.md"
    shutil.copyfile(protocol_path, out / "protocol.md")
    shutil.copyfile(Path(__file__), out / "as_run_guarded_component.py")
    data.to_csv(out / "development_input.csv", index=False)
    spec = dict(
        id=CANDIDATE,
        description="Training-guarded retained component with original-baseline fallback",
        features=previous["arms"][1]["features"],
        components=[dict(id="crossfit_components")],
    )
    protocol = dict(
        arms=[previous["arms"][0], spec],
        splits=previous["splits"],
        repeat_seeds=previous["repeat_seeds"],
        period_limits=LIMITS,
        interval_tail_probability=TAIL,
        alpha_spending_cycle=9,
        new_fits=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "preexecution_identity.json",
        dict(
            inputs=input_identity(
                {"source": source / "manifest.json", "protocol": protocol_path}
            ),
            implementation=input_identity(
                {
                    "runner": Path(__file__),
                    "assessment": ROOT / "comparisons/matched_evaluation.py",
                    "confirmation": ROOT / "comparisons/additional_uncertainty.py",
                    "simulation": ROOT / "comparisons/bootstrap_resolution.py",
                }
            ),
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
        inner = inner.merge(
            train[KEYS + ["n_orders_3m"]], on=KEYS, validate="many_to_one"
        )
        baseline = inner[inner.candidate.eq("baseline")]
        alternative = inner[inner.candidate.eq(COMPONENT)]
        selected, scan, groups = choose_component(baseline, alternative)
        write_json(folder / "selection.json", selected)
        scan.to_csv(folder / "inner_cutoff_scan.csv", index=False)
        groups.to_csv(folder / "reference_groups.csv", index=False)
        # The component/cutoff are frozen above, before loading any outer predictions.
        outer = read_csv(original / "predictions.csv")
        original_baseline = outer[outer.strategy.eq("baseline")]
        require(
            original_baseline.cutoff.eq(selected["reference_cutoff"]).all(),
            "Baseline cutoff changed",
        )
        chosen = outer[outer.strategy.eq(selected["component"])].assign(
            strategy=CANDIDATE, cutoff=selected["cutoff"]
        )
        predictions = pd.concat([original_baseline, chosen], ignore_index=True)
        chosen_inner = inner[inner.candidate.eq(selected["component"])].assign(
            candidate=CANDIDATE
        )
        saved_inner = pd.concat([baseline, chosen_inner], ignore_index=True)
        saved_inner.to_csv(folder / "inner_scores.csv", index=False)
        predictions.to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(
            [
                dict(candidate="baseline", cutoff=selected["reference_cutoff"]),
                dict(candidate=CANDIDATE, cutoff=selected["cutoff"]),
            ]
        ).to_csv(folder / "inner_selections.csv", index=False)
        shutil.copyfile(
            original / "model_identities.json", folder / "model_identities.json"
        )
        shutil.copyfile(
            original / "cached_hgb_identities.json",
            folder / "cached_hgb_identities.json",
        )
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
    pd.DataFrame(choices).to_csv(out / "component_choices.csv", index=False)
    pd.concat(transfers, ignore_index=True).to_csv(
        out / "cutoff_transfer.csv", index=False
    )
    child = out / CANDIDATE
    publish_assessment(child, out, data, profiles, predictions, protocol)
    decision = intersection_gate(child)
    write_json(
        child / "completion_manifest.json",
        dict(
            outputs={
                p.name: input_identity({"file": p})["file"]["sha256"]
                for p in child.iterdir()
                if p.is_file()
            }
        ),
    )
    final = additional_confirmation(child, CANDIDATE, TAIL)
    write_json(
        out / "final_procedure_decision.json",
        dict(
            primary=decision,
            final=final,
            selected_component_folds=sum(c["component"] == COMPONENT for c in choices),
        ),
    )
    write_manifest(
        out,
        dict(
            stage="guarded_component",
            fits=0,
            inputs=input_identity({"source": source / "manifest.json"}),
            accepted=final["accepted"],
        ),
    )
    print(
        pd.DataFrame(choices)[
            ["repeat", "outer_fold", "component", "reason"]
        ].to_string(index=False)
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

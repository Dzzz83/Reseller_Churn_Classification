"""Explain finite training cutoff infeasibility using exact monotone count bounds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.breadth_reporting import markdown_table
from comparisons.development_reporting import slices
from comparisons.guarded_component import COMPONENT, count_curve
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def bounds_for_partition(reference, candidate, reference_cutoff):
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    candidate = candidate.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        reference[KEYS + ["churn", "n_orders_3m"]],
        candidate[KEYS + ["churn", "n_orders_3m"]],
    )
    cutoffs = np.unique(np.r_[0.0, candidate.score])
    records = []
    for dimension, name, group in slices(reference):
        target = group.score.ge(reference_cutoff)
        positive = group.churn.eq(1)
        baseline_tp = int((target & positive).sum())
        baseline_fp = int((target & ~positive).sum())
        tp, fp = count_curve(candidate.loc[group.index], cutoffs)
        churners = int(positive.sum())
        require(churners > 0, "Undefined subgroup recall cannot define a cutoff bound")
        if dimension == "overall":
            required_tp = max((9 * churners) // 10 + 1, baseline_tp)
        else:
            required_tp = max(0, (20 * baseline_tp - churners + 19) // 20)
        upper = np.flatnonzero(tp >= required_tp)
        records.append(
            dict(
                slice=dimension,
                group=name,
                constraint="recall",
                rows=len(group),
                churners=churners,
                baseline_tp=baseline_tp,
                baseline_fp=baseline_fp,
                required_tp=required_tp,
                bound_index=int(upper[-1]) if len(upper) else -1,
                bound_cutoff=float(cutoffs[upper[-1]]) if len(upper) else np.nan,
            )
        )
        if dimension != "overall":
            lower = np.flatnonzero(fp <= baseline_fp)
            records.append(
                dict(
                    slice=dimension,
                    group=name,
                    constraint="fp",
                    rows=len(group),
                    churners=churners,
                    baseline_tp=baseline_tp,
                    baseline_fp=baseline_fp,
                    required_tp=required_tp,
                    bound_index=int(lower[0]) if len(lower) else len(cutoffs),
                    bound_cutoff=float(cutoffs[lower[0]]) if len(lower) else np.nan,
                )
            )
    rows = pd.DataFrame(records)
    fp_lower = int(rows.loc[rows.constraint.eq("fp"), "bound_index"].max())
    recall_upper = int(rows.loc[rows.constraint.eq("recall"), "bound_index"].min())
    binding_fp = rows[rows.constraint.eq("fp") & rows.bound_index.eq(fp_lower)]
    binding_recall = rows[
        rows.constraint.eq("recall") & rows.bound_index.eq(recall_upper)
    ]
    summary = dict(
        cutoff_count=len(cutoffs),
        fp_lower_index=fp_lower,
        recall_upper_index=recall_upper,
        feasible=fp_lower <= recall_upper,
        fp_required_cutoff=float(cutoffs[fp_lower])
        if fp_lower < len(cutoffs)
        else None,
        recall_max_cutoff=float(cutoffs[recall_upper]) if recall_upper >= 0 else None,
        binding_fp="; ".join(binding_fp["slice"] + ":" + binding_fp["group"]),
        binding_recall="; ".join(
            binding_recall["slice"] + ":" + binding_recall["group"]
        ),
    )
    # Independent direct feasibility count verifies the interval intersection.
    feasible = np.ones(len(cutoffs), dtype=bool)
    for row in records:
        indices = np.arange(len(cutoffs))
        feasible &= (
            indices >= row["bound_index"]
            if row["constraint"] == "fp"
            else indices <= row["bound_index"]
        )
    require(bool(feasible.any()) == summary["feasible"], "Bound intersection differs")
    return rows, summary


def run(output):
    source = ROOT / "results/guarded-component-20261018"
    components = ROOT / "results/retained-calibration-20261018"
    verify_manifest(source)
    verify_manifest(components)
    data = read_csv(source / "development_input.csv")
    protocol = json.loads((source / "protocol.json").read_text())
    all_bounds, summaries = [], []
    for split in protocol["splits"]:
        train, _ = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        inside = read_csv(components / relative / "inner_scores.csv").merge(
            train[KEYS + ["n_orders_3m"]], on=KEYS, validate="many_to_one"
        )
        choice = json.loads((source / relative / "selection.json").read_text())
        bounds, summary = bounds_for_partition(
            inside[inside.candidate.eq("baseline")],
            inside[inside.candidate.eq(COMPONENT)],
            choice["reference_cutoff"],
        )
        require(
            summary["feasible"] == (choice["component_feasible_cutoffs"] > 0),
            "Saved cutoff feasibility differs",
        )
        all_bounds.append(
            bounds.assign(repeat=split["repeat"], outer_fold=split["outer_fold"])
        )
        summaries.append(
            dict(repeat=split["repeat"], outer_fold=split["outer_fold"], **summary)
        )
    out = new_output_directory(output)
    bounds = pd.concat(all_bounds, ignore_index=True)
    summary = pd.DataFrame(summaries)
    bounds.to_csv(out / "all_training_bounds.csv", index=False)
    summary.to_csv(out / "binding_training_bounds.csv", index=False)
    (out / "report.md").write_text(
        "# Binding training cutoff constraints\n\n"
        "No model, calibrator or new operating rule was fitted. These are INNER training diagnostics, "
        "not outer-label cutoffs. Every admissible score tie is preserved.\n\n"
        + markdown_table(
            summary[
                [
                    "repeat",
                    "outer_fold",
                    "fp_required_cutoff",
                    "recall_max_cutoff",
                    "binding_fp",
                    "binding_recall",
                    "feasible",
                ]
            ]
        )
        + "\n\nFalse-alert limits require the cutoff to be at least the listed lower bound; recall limits "
        "require it to be no greater than the upper bound. A conflicting interval explains why this "
        "fixed component cannot satisfy all inner guards with one cutoff. It does not prove a causal "
        "calibration or population mechanism, rule out different representations, or justify relaxing "
        "any frozen acceptance requirement. All groups are retained in all_training_bounds.csv.\n"
    )
    write_manifest(
        out,
        dict(
            models_fitted=0,
            partitions=len(summaries),
            inputs=input_identity(
                {
                    "guarded": source / "manifest.json",
                    "components": components / "manifest.json",
                    "script": Path(__file__),
                }
            ),
        ),
    )
    print(
        summary[["repeat", "outer_fold", "binding_fp", "binding_recall"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

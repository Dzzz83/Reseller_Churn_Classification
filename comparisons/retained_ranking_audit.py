"""Descriptive tied-score ranking feasibility at baseline catches, without fitting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv
from comparisons.breadth_reporting import markdown_table
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)


def tied_burden(rows, target):
    if target == 0:
        return 0, 0
    grouped = (
        rows.groupby("score").churn.agg(["sum", "count"]).sort_index(ascending=False)
    )
    tp = grouped["sum"].cumsum()
    point = np.flatnonzero(tp.to_numpy() >= target)[0]
    return int(tp.iloc[point]), int(
        grouped["count"].cumsum().iloc[point] - tp.iloc[point]
    )


def run(source, output):
    verify_manifest(source)
    data = read_csv(source / "validation_predictions.csv")
    records = []
    for (repeat, snapshot), part in data.groupby(["repeat", "snapshot"]):
        baseline = part[part.strategy.eq("baseline")]
        candidate = part[part.strategy.ne("baseline")]
        target = int((baseline.churn.eq(1) & baseline.score.ge(baseline.cutoff)).sum())
        original_fp = int(
            (baseline.churn.eq(0) & baseline.score.ge(baseline.cutoff)).sum()
        )
        tp, fp = tied_burden(candidate, target)
        # Independent exhaustive enumeration, preserving every tied block.
        choices = [
            (
                int((candidate.churn.eq(1) & candidate.score.ge(c)).sum()),
                int((candidate.churn.eq(0) & candidate.score.ge(c)).sum()),
            )
            for c in candidate.score.unique()
        ]
        assert fp == min(f for t, f in choices if t >= target)
        records.append(
            dict(
                repeat=int(repeat),
                snapshot=str(snapshot.date()),
                baseline_tp=target,
                baseline_fp=original_fp,
                descriptive_candidate_tp=tp,
                descriptive_minimum_fp=fp,
                burden_difference=fp - original_fp,
            )
        )
    out = new_output_directory(output)
    rows = pd.DataFrame(records)
    rows.to_csv(out / "descriptive_burden.csv", index=False)
    (out / "report.md").write_text(
        "# Descriptive period ranking feasibility\n\n"
        "Outcome-derived points, never deployable cutoffs. Each period pools separate outer models. "
        "No fitted parameters or operating rules selected. All ties enter together.\n\n"
        + markdown_table(rows)
        + "\n"
    )
    write_manifest(
        out,
        {
            "inputs": input_identity(
                {"source": source / "manifest.json", "script": Path(__file__)}
            ),
            "models_fitted": 0,
            "exhaustive_tie_checks": len(rows),
        },
    )
    print(rows.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

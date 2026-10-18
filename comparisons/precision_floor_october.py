"""Freeze development precision-floor cutoffs, then evaluate all on inspected October."""

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.development_checks import validate_development_cases
from comparisons.evaluate_frozen_october import verified_artifact
from comparisons.positive_floor_october_evaluation import join_supplied_outcomes
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)
from reseller_churn.modeling.grouped_validation import ThresholdPolicy
from reseller_churn.modeling.grouped_validation import cutoff_scan as cutoff_scan

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/positive-floor-october-evaluation-20261018"
CANDIDATE = "positive_floor_veto"
FLOORS = list(range(20, 101, 10))


def select_floor(scan, minimum_percent):
    require(
        0 < minimum_percent <= 100 and int(minimum_percent) == minimum_percent,
        "Invalid precision floor",
    )
    return ThresholdPolicy(minimum_precision=minimum_percent / 100).select(scan)


def independent_choice(labels, scores, minimum_percent):
    options = []
    y = np.asarray(labels) == 1
    score = np.asarray(scores)
    for cut in sorted(set([0.0, *score])):
        alert = score >= cut
        tp = int((y & alert).sum())
        fp = int((~y & alert).sum())
        if tp + fp > 0 and 100 * tp >= minimum_percent * (tp + fp):
            options.append((tp, Fraction(tp, tp + fp), cut))
    return None if not options else float(max(options)[2])


def run(output):
    verify_manifest(SOURCE)
    out = new_output_directory(output)
    write_json(
        out / "protocol.json",
        dict(
            candidate=CANDIDATE,
            precision_floors_percent=FLOORS,
            authorization="User: select thresholds on minimum precision 20,30,40,... in development, then evaluate final test.",
            objective="Maximize pooled development OOF recall subject to precision >= floor; ties higher precision, then higher cutoff.",
            recall_floor=None,
            models_unchanged=True,
            veto_floor_unchanged=True,
            new_fits=0,
            infeasible="Report without a selected cutoff or test predictions; never silently drop.",
            undefined="No-alert precision is undefined and infeasible. All supplied cases retained.",
            test_status="Previously inspected October; descriptive reuse, no test-based winner selection.",
            inputs=input_identity(
                {"prepared_source": SOURCE / "manifest.json", "script": Path(__file__)}
            ),
        ),
    )
    (out / "as_run_precision_floor_october.py").write_bytes(Path(__file__).read_bytes())
    inside = read_csv(SOURCE / "development_oof.csv")
    inside = inside[inside.candidate.eq(CANDIDATE)].copy()
    validate_development_cases(inside)
    require(
        len(inside) == 1375
        and inside.churn.sum() == 212
        and inside.StoreID.nunique() == 422,
        "Development population changed",
    )
    scan = cutoff_scan(inside.churn, inside.score)
    scan.to_csv(out / "development_cutoff_scan.csv", index=False)
    selections = []
    for floor in FLOORS:
        selected = select_floor(scan, floor)
        direct = independent_choice(inside.churn, inside.score, floor)
        require(
            (selected is None and direct is None)
            or (selected is not None and selected["cutoff"] == direct),
            "Independent selection differs",
        )
        selections.append(
            dict(
                minimum_precision_percent=floor,
                feasible=selected is not None,
                **(
                    {
                        k: selected[k]
                        for k in [
                            "cutoff",
                            "tp",
                            "fp",
                            "fn",
                            "tn",
                            "recall",
                            "precision",
                            "alerts",
                        ]
                    }
                    if selected is not None
                    else {}
                ),
            )
        )
    choices = pd.DataFrame(selections)
    choices.to_csv(out / "development_selections.csv", index=False)
    # All cutoffs are fixed before opening supplied test outcomes.
    test = read_csv(SOURCE / "unlabeled_predictions.csv")
    test = test[test.strategy.eq(CANDIDATE)].copy()
    parts = []
    for choice in selections:
        if not choice["feasible"]:
            continue
        part = test.assign(
            strategy="precision_floor_" + str(choice["minimum_precision_percent"]),
            cutoff=choice["cutoff"],
        )
        part["predicted_churn"] = part.score.ge(part.cutoff).astype(int)
        parts.append(part)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            out / "unlabeled_predictions.csv", index=False
        )
    write_manifest(
        out,
        dict(
            stage="precision_floor_predictions_checkpointed",
            outcomes_read=False,
            models_fitted=0,
        ),
    )
    verify_manifest(out)
    evaluation = new_output_directory(out / "evaluation")
    if parts:
        path = ROOT / "results/current-model-diagnosis/october_cases.csv"
        verified_artifact(path.parent / "manifest.json", path)
        labels = pd.read_csv(path, usecols=study.KEYS + ["churn"])
        cases = join_supplied_outcomes(pd.concat(parts, ignore_index=True), labels)
        cases.to_csv(evaluation / "evaluated_cases.csv", index=False)
        metrics = []
        for choice in selections:
            if not choice["feasible"]:
                continue
            policy = "precision_floor_" + str(choice["minimum_precision_percent"])
            part = cases[cases.strategy.eq(policy)]
            for group, rows in [
                ("overall", part),
                ("recent_purchase", part[part.n_orders_3m.gt(0)]),
                ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
            ]:
                m = study.decision_metrics(rows.churn, rows.score, rows.cutoff)
                y = rows.churn.eq(1)
                alert = rows.predicted_churn.eq(1)
                require(
                    [m[k] for k in ["tp", "fp", "fn", "tn"]]
                    == [
                        int((y & alert).sum()),
                        int((~y & alert).sum()),
                        int((y & ~alert).sum()),
                        int((~y & ~alert).sum()),
                    ],
                    "Independent confusion differs",
                )
                metrics.append(
                    dict(
                        minimum_precision_percent=choice["minimum_precision_percent"],
                        subgroup=group,
                        precision_defined=bool(alert.any()),
                        recall_defined=bool(y.any()),
                        **m,
                    )
                )
        metric_frame = pd.DataFrame(metrics)
        metric_frame.to_csv(evaluation / "metrics.csv", index=False)
        overall = metric_frame[metric_frame.subgroup.eq("overall")]
        summary = choices.merge(
            overall.drop(
                columns=[
                    "subgroup",
                    "precision_defined",
                    "recall_defined",
                    "rows",
                    "churners",
                    "average_precision",
                ]
            ),
            on="minimum_precision_percent",
            how="left",
            suffixes=("_development", "_test"),
        )
    else:
        summary = choices
    summary.to_csv(evaluation / "comparison.csv", index=False)
    report = (
        """# Development precision-floor thresholds evaluated on inspected October

Model and veto floor unchanged. For each predeclared target20%,30%,...,100%,
maximize development OOF recall subject to precision >=target; ties higher
precision then higher cutoff. This replaces the earlier recall>90% objective;
there is NO recall floor here. All chosen cutoffs were frozen and unlabeled test
predictions checkpointed before supplied outcomes. Zero new model fits.
Development is1375rows/212churners/422resellers; October492rows/46churners.

A development precision target is NOT a guarantee of test precision. These are
in-sample threshold-selection metrics on development OOF scores; fitting models
for OOF is isolated, but choosing/reporting cutoffs on these outcomes is adaptive.
October was previously inspected; the requested sweep is descriptive reuse,
not independent validation or authorization to select a test-winning policy.
All targets, infeasible settings, periods/activity groups and counts retained.
Earlier full development acceptance decisions remain unchanged. No deployment.

```text
"""
        + summary.to_string(index=False)
        + "\n```\n"
    )
    (evaluation / "report.md").write_text(report)
    write_manifest(
        evaluation,
        dict(
            stage="all_precision_floors_evaluated",
            models_fitted=0,
            cutoffs_selected_on_test=0,
            inputs=input_identity({"prediction_checkpoint": out / "manifest.json"}),
            independent_selection_checks=len(FLOORS),
            independent_confusion_checks=len(metrics) if parts else 0,
        ),
    )
    verify_manifest(evaluation)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

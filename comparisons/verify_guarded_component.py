"""Independent exhaustive training-cutoff and fallback audit; no model fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.verify_basket_breadth import independent_cutoff
from comparisons.verify_retained_calibration import confusion, ordered
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
COMPONENT = "retained_logistic"
CANDIDATE = "guarded_component"


def enumerate_cutoffs(reference, candidate, baseline_cutoff):
    """Separate direct >= enumeration, with integer subgroup count arithmetic."""
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    candidate = candidate.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        reference[KEYS + ["churn", "n_orders_3m"]],
        candidate[KEYS + ["churn", "n_orders_3m"]],
    )
    y = reference.churn.to_numpy() == 1
    a = reference.score.to_numpy() >= baseline_cutoff
    score = candidate.score.to_numpy()
    recent = reference.n_orders_3m.to_numpy() > 0
    masks = [recent, ~recent]
    for snapshot in sorted(reference.snapshot.unique()):
        mask = reference.snapshot.eq(snapshot).to_numpy()
        masks.extend([mask, mask & recent, mask & ~recent])
    groups = [(np.flatnonzero(y & mask), np.flatnonzero(~y & mask)) for mask in masks]
    undefined = any(len(pos) == 0 for pos, _ in groups)
    records = []
    for cutoff in sorted(set([0.0, *score])):
        alert = score >= cutoff
        tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
        feasible = tp / y.sum() > 0.90 and tp >= int((a & y).sum()) and not undefined
        for pos, neg in groups:
            # max_recall_loss=.05 is the exact rational 1/20.
            feasible &= (int(alert[pos].sum()) - int(a[pos].sum())) * 20 >= -len(pos)
            feasible &= alert[neg].sum() <= a[neg].sum()
        records.append(
            dict(
                cutoff=float(cutoff),
                tp=tp,
                fp=fp,
                recall=tp / int(y.sum()),
                precision=tp / max(1, tp + fp),
                feasible=bool(feasible),
            )
        )
    eligible = [row for row in records if row["feasible"]]
    best = (
        max(eligible, key=lambda row: (row["precision"], row["recall"], row["cutoff"]))
        if eligible
        else None
    )
    return best, records, undefined


def independent_choice(reference, candidate):
    reference_cutoff = independent_cutoff(
        reference.churn.to_numpy(), reference.score.to_numpy()
    )
    baseline_best, baseline_scan, undefined = enumerate_cutoffs(
        reference, reference, reference_cutoff
    )
    best, alternative_scan, _ = enumerate_cutoffs(
        reference, candidate, reference_cutoff
    )
    reference_fp = int(
        (reference.churn.eq(0) & reference.score.ge(reference_cutoff)).sum()
    )
    if undefined:
        best = None
        reason = "undefined_inner_group"
    else:
        require(
            baseline_best is not None and baseline_best["cutoff"] == reference_cutoff,
            "Constrained baseline differs",
        )
        reason = (
            "no_feasible_component_cutoff"
            if best is None
            else "inner_practical_gain_not_met"
        )
    select = (
        best is not None
        and best["fp"] < reference_fp
        and (reference_fp - best["fp"]) * 20 >= reference_fp
    )
    if select:
        reason = "inner_guards_and_practical_gain_passed"
    choice = dict(
        component=COMPONENT if select else "baseline",
        cutoff=best["cutoff"] if select else reference_cutoff,
        reason=reason,
        reference_cutoff=reference_cutoff,
        reference_fp=reference_fp,
        component_feasible_cutoffs=sum(row["feasible"] for row in alternative_scan),
        component_cutoff=None if best is None else best["cutoff"],
        component_fp=None if best is None else best["fp"],
    )
    return choice, {"baseline": baseline_scan, COMPONENT: alternative_scan}


def verify(directory):
    verify_manifest(directory)
    pre = json.loads((directory / "preexecution_identity.json").read_text())
    for identity in [*pre["inputs"].values(), *pre["implementation"].values()]:
        path = (
            directory / "as_run_guarded_component.py"
            if Path(identity["path"]).name == "guarded_component.py"
            else Path(identity["path"])
        )
        require(
            file_sha256(path) == identity["sha256"], "Pre-execution identity changed"
        )
    source = directory.parent / "retained-calibration-20261018"
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    complete = json.loads(
        (directory / CANDIDATE / "completion_manifest.json").read_text()
    )
    for name, digest in complete["outputs"].items():
        require(
            file_sha256(directory / CANDIDATE / name) == digest,
            "Final primary assessment changed",
        )
    choices, boundaries = [], 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        original, folder = source / relative, directory / relative
        verify_manifest(original)
        verify_manifest(folder)
        inside = read_csv(original / "inner_scores.csv").merge(
            train[KEYS + ["n_orders_3m"]], on=KEYS, validate="many_to_one"
        )
        baseline, candidate = (
            inside[inside.candidate.eq("baseline")],
            inside[inside.candidate.eq(COMPONENT)],
        )
        expected, scans = independent_choice(baseline, candidate)
        require(
            expected == json.loads((folder / "selection.json").read_text()),
            "Training component/cutoff choice differs",
        )
        saved_scan = pd.read_csv(
            folder / "inner_cutoff_scan.csv", float_precision="round_trip"
        )
        for component, records in scans.items():
            saved = saved_scan[saved_scan.component.eq(component)].sort_values("cutoff")
            direct = pd.DataFrame(records).sort_values("cutoff")
            for name in ["cutoff", "tp", "fp", "recall", "precision", "feasible"]:
                np.testing.assert_array_equal(
                    saved[name].to_numpy(), direct[name].to_numpy()
                )
            boundaries += len(records)
        # Selection is already independently fixed before outer outcomes enter this audit.
        original_outer = read_csv(original / "predictions.csv")
        predictions = read_csv(folder / "predictions.csv")
        selected = original_outer[
            original_outer.strategy.eq(expected["component"])
        ].assign(strategy=CANDIDATE, cutoff=expected["cutoff"])
        pd.testing.assert_frame_equal(
            ordered(selected), ordered(predictions[predictions.strategy.eq(CANDIDATE)])
        )
        pd.testing.assert_frame_equal(
            ordered(original_outer[original_outer.strategy.eq("baseline")]),
            ordered(predictions[predictions.strategy.eq("baseline")]),
        )
        for _, rows in predictions.groupby("strategy"):
            pd.testing.assert_frame_equal(
                ordered(rows[KEYS + ["churn", "n_orders_3m"]]),
                ordered(valid[KEYS + ["churn", "n_orders_3m"]]),
            )
        choices.append(expected)
    predictions = read_csv(directory / "validation_predictions.csv")
    metrics = pd.read_csv(directory / CANDIDATE / "operating_metrics.csv")
    for _, row in metrics.iterrows():
        rows = predictions[
            predictions.strategy.eq(row.strategy) & predictions.repeat.eq(row["repeat"])
        ]
        if row.scope == "outer_fold":
            rows = rows[rows.outer_fold.eq(row.outer_fold)]
        if row["slice"] in ["snapshot", "snapshot_activity"]:
            rows = rows[rows.snapshot.eq(row["group"].split("/")[0])]
        if row["slice"] in ["activity", "snapshot_activity"]:
            rows = rows[
                rows.n_orders_3m.gt(0).eq(
                    row["group"].split("/")[-1] == "recent_purchase"
                )
            ]
        require(
            confusion(rows) == [row[name] for name in ["tp", "fp", "fn", "tn"]],
            "Operating count differs",
        )
    result = dict(
        status="passed",
        component_choices=len(choices),
        exhaustive_tied_boundaries=boundaries,
        selected_component_folds=sum(row["component"] == COMPONENT for row in choices),
        operating_rows=len(metrics),
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(
        not (directory / "independent_verification.json").exists(),
        "Preserve verification",
    )
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

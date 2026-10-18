"""Reconstruct every fixed blend/control score and prior from saved development data."""

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
CANDIDATE = "complementary_mean"
CONTROL = "corrected_experts_control"


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    require(
        not protocol["control"]["eligible_for_selection"], "Control cannot be a winner"
    )
    child = directory / CANDIDATE
    completion = json.loads((child / "completion_manifest.json").read_text())
    for name, digest in completion["outputs"].items():
        require(
            file_sha256(child / name) == digest, "Completed assessment checksum changed"
        )
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        path = Path(identity["path"])
        if path.name == "continuation_blend.py":
            path = directory / "as_run_continuation_blend.py"
        require(
            file_sha256(path) == identity["sha256"], "Pre-execution code/input changed"
        )
    calibrated = directory.parent / "retained-calibration-20261018"
    experts = directory.parent / "activity-experts-20261018-02"
    data = read_csv(directory / "development_input.csv")
    checked = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        original_inner = read_csv(calibrated / relative / "inner_scores.csv")
        original_expert_inner = read_csv(experts / relative / "inner_scores.csv")
        for stage in ["outer", 0, 1, 2]:
            if stage == "outer":
                fitting, held = train, valid
                a = read_csv(calibrated / relative / "predictions.csv")
                b = read_csv(experts / relative / "predictions.csv")
                a = a[a.strategy.eq("retained_logistic")]
                b = b[b.strategy.eq("activity_experts")]
                saved = outer
                column = "strategy"
            else:
                member = inner[
                    inner.candidate.eq("baseline") & inner.inner_fold.eq(stage)
                ]
                fitting = train[~train.StoreID.isin(member.StoreID)]
                held = train[train.StoreID.isin(member.StoreID)]
                a = original_inner[
                    original_inner.candidate.eq("retained_logistic")
                    & original_inner.inner_fold.eq(stage)
                ]
                b = original_expert_inner[
                    original_expert_inner.candidate.eq("activity_experts")
                    & original_expert_inner.inner_fold.eq(stage)
                ]
                saved = inner[inner.inner_fold.eq(stage)]
                column = "candidate"
            require(set(fitting.StoreID).isdisjoint(held.StoreID), "Prior leakage")
            p = float(fitting.churn.sum() / len(fitting))
            base = held[KEYS + ["churn"]].merge(
                a[KEYS + ["churn", "score"]], on=KEYS + ["churn"], validate="one_to_one"
            )
            base = base.rename(columns={"score": "calibration_score"}).merge(
                b[KEYS + ["churn", "score"]], on=KEYS + ["churn"], validate="one_to_one"
            )
            q = base.score.to_numpy()
            corrected = q / (q + (1 - q) * (1 - p) / p)
            mean = (base.calibration_score.to_numpy() + corrected) / 2
            for arm, expected in [(CONTROL, corrected), (CANDIDATE, mean)]:
                rows = base[KEYS + ["churn"]].merge(
                    saved[saved[column].eq(arm)],
                    on=KEYS + ["churn"],
                    validate="one_to_one",
                )
                require(len(rows) == len(held), "Incomplete stage population")
                np.testing.assert_allclose(rows.score, expected, atol=2e-16, rtol=1e-13)
                checked += 1
        for arm in ["baseline", CANDIDATE, CONTROL]:
            inside = inner[inner.candidate.eq(arm)]
            require(
                outer[outer.strategy.eq(arm)]
                .cutoff.eq(independent_cutoff(inside.churn, inside.score))
                .all(),
                "Cutoff differs",
            )
        cached = read_csv(calibrated / relative / "predictions.csv")
        pd.testing.assert_frame_equal(
            ordered(outer[outer.strategy.eq("baseline")]),
            ordered(cached[cached.strategy.eq("baseline")]),
        )
    predictions = read_csv(directory / "validation_predictions.csv")
    count = 0
    for metrics_path in [
        child / "operating_metrics.csv",
        directory / CONTROL / "operating_metrics.csv",
    ]:
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            rows = predictions[
                predictions.strategy.eq(row.strategy)
                & predictions.repeat.eq(row["repeat"])
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
                confusion(rows) == [row[k] for k in ["tp", "fp", "fn", "tn"]],
                "Operating counts differ",
            )
            count += 1
    result = dict(
        status="passed",
        score_stage_constructions=checked,
        cutoffs=45,
        operating_rows=count,
        classifiers_fitted=0,
        calibrators_fitted=0,
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

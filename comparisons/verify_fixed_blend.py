"""Reconstruct a fixed blend and every inner cutoff independently, without fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    features = read_csv(directory / "development_input.csv")
    predictions = read_csv(directory / "validation_predictions.csv")
    candidate = "fixed_equal_blend"
    require(len(predictions) == 8250, "Incomplete paired study")
    checked = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(features, split)
        repeat, fold = split["repeat"], split["outer_fold"]
        folder = directory / f"repeat_{repeat}/fold_{fold}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        for kind, filename, sourcefile, groupcol in [
            ("inner", "inner_scores.csv", "component_inner_scores.csv", "candidate"),
            ("outer", "predictions.csv", "component_outer_scores.csv", "strategy"),
        ]:
            actual = read_csv(folder / filename)
            component = read_csv(folder / sourcefile)
            baseline = (
                component[component[groupcol].eq("baseline")]
                .sort_values(["snapshot", "StoreID"])
                .reset_index(drop=True)
            )
            cadence = (
                component[component[groupcol].eq("cadence_no_count")]
                .sort_values(["snapshot", "StoreID"])
                .reset_index(drop=True)
            )
            blended = (
                actual[actual[groupcol].eq(candidate)]
                .sort_values(["snapshot", "StoreID"])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                baseline[KEYS + ["churn"]], cadence[KEYS + ["churn"]]
            )
            pd.testing.assert_frame_equal(
                blended[KEYS + ["churn"]], baseline[KEYS + ["churn"]]
            )
            np.testing.assert_allclose(
                blended.score, (baseline.score + cadence.score) / 2, rtol=0, atol=0
            )
            population = train if kind == "inner" else valid
            pd.testing.assert_frame_equal(
                blended[KEYS + ["churn"]],
                population.sort_values(["snapshot", "StoreID"])[
                    KEYS + ["churn"]
                ].reset_index(drop=True),
            )
            if kind == "inner":
                require(
                    baseline.inner_fold.eq(cadence.inner_fold).all(),
                    "Component inner partitions differ",
                )
                require(
                    baseline.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                    "Inner reseller leakage",
                )
        for arm in ("baseline", candidate):
            part = inner[inner.candidate.eq(arm)]
            cutoff = independent_cutoff(part.churn.to_numpy(), part.score.to_numpy())
            saved = predictions[
                (predictions.repeat == repeat)
                & (predictions.outer_fold == fold)
                & predictions.strategy.eq(arm)
            ]
            require(saved.cutoff.eq(cutoff).all(), "Cutoff is not inner selected")
            checked += 1
    metrics = pd.read_csv(directory / "metrics.csv")
    overall = metrics[(metrics.scope == "repeat_pooled") & metrics.slice.eq("overall")]
    for row in overall.itertuples(index=False):
        part = predictions[
            predictions.repeat.eq(row.repeat) & predictions.strategy.eq(row.policy)
        ]
        a, y = part.score.ge(part.cutoff), part.churn.eq(1)
        require(
            [int(x.sum()) for x in [a & y, a & ~y, ~a & y, ~a & ~y]]
            == [row.tp, row.fp, row.fn, row.tn],
            "Confusion counts differ",
        )
    result = {
        "status": "passed",
        "outer_splits": 15,
        "inner_cutoffs_independently_reproduced": checked,
        "exact_score_constructions_verified": 30,
        "overall_repeat_metric_rows": len(overall),
        "models_fitted": 0,
        "october_outcomes_accessed": False,
    }
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

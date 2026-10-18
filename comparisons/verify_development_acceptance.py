"""Independently recount prospective matched development acceptance, no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require
from reseller_churn.artifacts import input_identity, write_json

KEYS = ["StoreID", "snapshot", "repeat", "outer_fold", "churn", "n_orders_3m"]


def verify_decision(directory, candidate, tail):
    predictions = read_csv(directory / "validation_predictions.csv")
    paired = predictions[predictions.strategy.eq("baseline")].merge(
        predictions[predictions.strategy.eq(candidate)],
        on=KEYS,
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    require(len(paired) == 4125, "Missing canonical cases/repeats")
    for arm in ["base", "new"]:
        paired["tp_" + arm] = paired["score_" + arm].ge(
            paired["cutoff_" + arm]
        ) & paired.churn.eq(1)
        paired["fp_" + arm] = paired["score_" + arm].ge(
            paired["cutoff_" + arm]
        ) & paired.churn.eq(0)
    pooled = paired.groupby("repeat")[
        ["tp_base", "tp_new", "fp_base", "fp_new", "churn"]
    ].sum()
    checks = {
        "recall_preserved_every_repeat": bool(pooled.tp_new.ge(pooled.tp_base).all()),
        "fewer_fp_every_repeat": bool(pooled.fp_new.lt(pooled.fp_base).all()),
        "at_least_five_percent_mean_fp_reduction": bool(
            (pooled.fp_base - pooled.fp_new).mean() / pooled.fp_base.mean() >= 0.05
        ),
    }
    slices = []
    for repeat, part in paired.groupby("repeat"):
        for dimension in ["activity", "snapshot", "snapshot_activity"]:
            columns = {
                "activity": ["n_orders_3m"],
                "snapshot": ["snapshot"],
                "snapshot_activity": ["snapshot", "n_orders_3m"],
            }[dimension]
            part = part.copy()
            part["activity"] = part.n_orders_3m.gt(0)
            columns = ["activity" if k == "n_orders_3m" else k for k in columns]
            for identity, group in part.groupby(columns):
                y = group.churn.sum()
                recall_change = (
                    (group.tp_new.sum() - group.tp_base.sum()) / y if y else np.nan
                )
                slices.append(
                    dict(
                        repeat=repeat,
                        dimension=dimension,
                        group=str(identity),
                        recall_change=recall_change,
                        fp_reduction=int(group.fp_base.sum() - group.fp_new.sum()),
                    )
                )
    slices = pd.DataFrame(slices)
    primary = slices[slices.dimension.isin(["snapshot", "activity"])]
    checks["frozen_period_activity_limits"] = bool(
        (primary.recall_change.ge(-0.05) & primary.fp_reduction.ge(0)).all()
    )
    intersections = slices[slices.dimension.eq("snapshot_activity")]
    checks["frozen_intersection_limits"] = bool(
        (intersections.recall_change.ge(-0.05) & intersections.fp_reduction.ge(0)).all()
    )
    checks["activity_snapshot_mean_recall_loss_at_most_five_points"] = bool(
        primary.groupby(["dimension", "group"]).recall_change.mean().ge(-0.05).all()
    )
    stores = np.sort(paired.StoreID.unique())
    weights = np.random.default_rng(2026).multinomial(
        len(stores), np.repeat(1 / len(stores), len(stores)), size=5000
    )
    deltas, reductions = [], []
    for _, part in paired.groupby("repeat"):
        grouped = (
            part.groupby("StoreID")[["tp_base", "tp_new", "fp_base", "fp_new", "churn"]]
            .sum()
            .reindex(stores, fill_value=0)
        )
        sampled = weights @ grouped.to_numpy(dtype=float)
        deltas.append((sampled[:, 1] - sampled[:, 0]) / sampled[:, 4])
        reductions.append(sampled[:, 2] - sampled[:, 3])
    recall = np.quantile(np.mean(deltas, axis=0), [tail, 1 - tail])
    fp = np.quantile(np.mean(reductions, axis=0), [tail, 1 - tail])
    checks["recall_uncertainty_lower_nonnegative"] = bool(recall[0] >= 0)
    checks["fp_reduction_uncertainty_lower_positive"] = bool(fp[0] > 0)
    decision = json.loads((directory / "decision.json").read_text())
    require(checks == decision["checks"], "Acceptance checks differ")
    require(all(checks.values()) == decision["accepted"], "Acceptance differs")
    np.testing.assert_array_equal(recall, decision["recall_difference_interval"])
    np.testing.assert_array_equal(fp, decision["fp_reduction_interval"])
    result = dict(
        status="passed",
        accepted=all(checks.values()),
        checks=checks,
        period_cells=len(slices),
        bootstrap_draws=5000,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    destination = directory / "acceptance_verification.json"
    require(not destination.exists(), "Preserve previous acceptance audit")
    write_json(destination, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tail", type=float, required=True)
    args = parser.parse_args()
    verify_decision(args.directory, args.candidate, args.tail)

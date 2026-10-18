"""Audit routed training identities, full weights and saved component score assembly."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.grouped_precision_recall import BASE_PARAMETERS, FEATURES, MODEL_SEEDS
from comparisons.verify_basket_breadth import independent_cutoff
from comparisons.verify_retained_calibration import confusion, ordered
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
CANDIDATE = "activity_experts"


def digest(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(
            frame[KEYS + ["churn"] + FEATURES], index=False
        ).values.tobytes()
    ).hexdigest()


def verify(directory):
    manifest = verify_manifest(directory)
    require(
        manifest["new_fits"] == 600 and manifest["cached_fits"] == 300,
        "Fit counts differ",
    )
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        archived = directory / "source_snapshot" / Path(identity["path"]).name
        path = archived if archived.is_file() else Path(identity["path"])
        require(file_sha256(path) == identity["sha256"], "Recorded input/code changed")
    data = read_csv(directory / "development_input.csv")
    protocol = json.loads((directory / "protocol.json").read_text())
    predictions = read_csv(directory / "validation_predictions.csv")
    source = directory.parent / "cadence-stability-20261016"
    original = read_csv(source / "validation_predictions.csv")
    pd.testing.assert_frame_equal(
        ordered(predictions[predictions.strategy.eq("baseline")]),
        ordered(original[original.strategy.eq("baseline")]),
    )
    checked = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        folder = directory / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        identities = json.loads((folder / "model_identities.json").read_text())
        for stage in ["outer", 0, 1, 2]:
            if stage == "outer":
                fitting, held = train, valid
                records = [
                    r
                    for r in identities
                    if r["candidate"] == CANDIDATE and r["stage"] == "outer"
                ]
                saved = outer[outer.strategy.eq(CANDIDATE)]
            else:
                member = inner[
                    inner.candidate.eq("baseline") & inner.inner_fold.eq(stage)
                ]
                fitting, held = (
                    train[~train.StoreID.isin(member.StoreID)],
                    train[train.StoreID.isin(member.StoreID)],
                )
                records = [
                    r
                    for r in identities
                    if r["candidate"] == CANDIDATE
                    and r["stage"] == "inner"
                    and r["inner_fold"] == stage
                ]
                saved = inner[
                    inner.candidate.eq(CANDIDATE) & inner.inner_fold.eq(stage)
                ]
            require(set(fitting.StoreID).isdisjoint(held.StoreID), "Reseller leakage")
            labels = fitting.churn.to_numpy()
            weights = len(fitting) / (2 * np.bincount(labels)[labels])
            require(len(records) == 10, "Expert component count")
            assembled = []
            for seed in MODEL_SEEDS:
                score = np.full(len(held), np.nan)
                for route in [False, True]:
                    record = next(
                        r
                        for r in records
                        if r["seed"] == seed and r["recent_purchase"] == route
                    )
                    mask = fitting.n_orders_3m.gt(0).eq(route).to_numpy()
                    held_mask = held.n_orders_3m.gt(0).eq(route).to_numpy()
                    require(
                        record["training_sha256"] == digest(fitting)
                        and record["expert_training_sha256"]
                        == digest(fitting.loc[mask]),
                        "Expert fitting identity",
                    )
                    np.testing.assert_array_equal(weights[mask], record["weights"])
                    expected_features = [
                        name
                        for name in FEATURES
                        if fitting.loc[mask, name].notna().any()
                    ]
                    require(
                        record["model_features"] == expected_features,
                        "Expert feature omission differs",
                    )
                    require(
                        record["omitted_all_missing"]
                        == [name for name in FEATURES if name not in expected_features],
                        "Omitted feature metadata differs",
                    )
                    keys = pd.DataFrame(record["validation_cases"], columns=KEYS)
                    keys["snapshot"] = pd.to_datetime(keys.snapshot)
                    pd.testing.assert_frame_equal(
                        keys, held.loc[held_mask, KEYS].reset_index(drop=True)
                    )
                    require(
                        all(
                            record["parameters"][k] == v
                            for k, v in BASE_PARAMETERS.items()
                        ),
                        "Model settings changed",
                    )
                    require(
                        record["parameters"]["random_state"] == seed, "Seed differs"
                    )
                    score[held_mask] = record["validation_scores"]
                    checked += 1
                require(np.isfinite(score).all(), "Missing routed output")
                assembled.append(score)
            aligned = held[KEYS].merge(saved, on=KEYS, validate="one_to_one")
            np.testing.assert_array_equal(aligned.score, np.mean(assembled, axis=0))
        for arm in ["baseline", CANDIDATE]:
            inside = inner[inner.candidate.eq(arm)]
            require(
                outer[outer.strategy.eq(arm)]
                .cutoff.eq(independent_cutoff(inside.churn, inside.score))
                .all(),
                "Cutoff differs",
            )
    metrics = pd.read_csv(directory / "operating_metrics.csv")
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
            confusion(rows) == [row[k] for k in ["tp", "fp", "fn", "tn"]],
            "Operating count mismatch",
        )
    result = dict(
        status="passed",
        expert_identities=checked,
        cutoffs=30,
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

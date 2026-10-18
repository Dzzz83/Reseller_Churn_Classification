"""Recount every saved training weight without fitting or model loading."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.grouped_precision_recall import BASE_PARAMETERS, MODEL_SEEDS
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for record in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        original = Path(record["path"])
        archived = directory / "source_snapshot" / original.name
        path = archived if archived.is_file() else original
        require(file_sha256(path) == record["sha256"], "Code/input changed")
    data = read_csv(directory / "development_input.csv")
    checked = 0
    totals = []
    for split in protocol["splits"]:
        train, _ = validate_partition(data, split)
        folder = directory / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        identities = json.loads((folder / "model_identities.json").read_text())
        for record in [r for r in identities if r["candidate"] == "reseller_weighted"]:
            fitting = train
            if record["stage"] == "inner":
                held = inner[
                    inner.candidate.eq("baseline")
                    & inner.inner_fold.eq(record["inner_fold"])
                ]
                fitting = train[~train.StoreID.isin(held.StoreID)]
            # Independent scalar formulation, not the training weight helper.
            counts = fitting.StoreID.value_counts().to_dict()
            inverse = np.array([1 / counts[i] for i in fitting.StoreID])
            labels = fitting.churn.to_numpy()
            class_totals = {label: sum(inverse[labels == label]) for label in [0, 1]}
            weights = np.array(
                [
                    v * len(fitting) / (2 * class_totals[y])
                    for v, y in zip(inverse, labels)
                ]
            )
            np.testing.assert_allclose(
                record["weights"], weights, atol=1e-14, rtol=1e-14
            )
            actual = np.array(record["weights"], dtype=float)
            require(
                hashlib.sha256(actual.tobytes()).hexdigest() == record["weight_sha256"],
                "Weight checksum changed",
            )
            for label in [0, 1]:
                np.testing.assert_allclose(
                    actual[labels == label].sum(), len(fitting) / 2, atol=1e-10
                )
            require(record["seed"] in MODEL_SEEDS, "Unknown model seed")
            for k, v in BASE_PARAMETERS.items():
                require(record["parameters"][k] == v, "Unplanned learner parameter")
            totals.append(
                dict(
                    training_rows=len(fitting),
                    class_zero_weight=float(actual[labels == 0].sum()),
                    class_one_weight=float(actual[labels == 1].sum()),
                    minimum_weight=float(actual.min()),
                    maximum_weight=float(actual.max()),
                )
            )
            checked += 1
    require(checked == 300, "Incomplete weight fits")
    result = dict(
        status="passed",
        weight_identities=checked,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(not (directory / "weight_verification.json").exists(), "Preserve old audit")
    write_json(directory / "weight_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

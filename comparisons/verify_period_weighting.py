"""Independently reconstruct all eight fitting-cell weights without refits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    data = read_csv(directory / "development_input.csv")
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for record in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        archived = directory / "source_snapshot" / Path(record["path"]).name
        path = archived if archived.is_file() else Path(record["path"])
        require(file_sha256(path) == record["sha256"], "Input/code checksum differs")
    checked = 0
    for split in protocol["splits"]:
        train, _ = validate_partition(data, split)
        folder = directory / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        records = json.loads((folder / "model_identities.json").read_text())
        for record in [r for r in records if r["candidate"] == "period_class_weighted"]:
            fitting = train
            if record["stage"] == "inner":
                held = inner[
                    inner.candidate.eq("baseline")
                    & inner.inner_fold.eq(record["inner_fold"])
                ]
                fitting = train[~train.StoreID.isin(held.StoreID)]
            weights = np.array(record["weights"], dtype=float)
            expected = np.zeros(len(fitting))
            cells = 0
            for snapshot in sorted(fitting.snapshot.unique()):
                for label in [0, 1]:
                    cell = (
                        fitting.snapshot.eq(snapshot).to_numpy()
                        & fitting.churn.eq(label).to_numpy()
                    )
                    require(cell.any(), "Missing fitting cell")
                    expected[cell] = len(fitting) / (8 * cell.sum())
                    np.testing.assert_allclose(
                        weights[cell].sum(), len(fitting) / 8, atol=1e-10
                    )
                    cells += 1
            require(cells == 8, "Wrong number of cells")
            np.testing.assert_array_equal(weights, expected)
            require(
                hashlib.sha256(weights.tobytes()).hexdigest()
                == record["weight_sha256"],
                "Weight bytes changed",
            )
            checked += 1
    require(checked == 300, "Missing planned fits")
    result = dict(
        status="passed",
        weight_identities=checked,
        eight_cell_normalization=True,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(not (directory / "weight_verification.json").exists(), "Preserve audit")
    write_json(directory / "weight_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

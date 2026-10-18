"""Verify the historical feature and final-stage lifecycle without model fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.bootstrap_resolution import resolution_plan
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
FEATURE = "early_quarter_order_share"
TAIL = 0.000001953125


def verify(directory):
    verify_manifest(directory)
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        archived = directory / "source_snapshot" / Path(identity["path"]).name
        path = archived if archived.is_file() else Path(identity["path"])
        require(file_sha256(path) == identity["sha256"], "Pre-fit code/input changed")
    data = read_csv(directory / "development_input.csv")
    source = directory.parent / "quarter-history-independent-audit-20261018"
    verify_manifest(source)
    independent = read_csv(source / "independent_features.csv")
    aligned = data[KEYS + [FEATURE]].merge(
        independent[KEYS + [FEATURE]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("_fit", "_independent"),
    )
    require(len(aligned) == 1375, "Feature population changed")
    np.testing.assert_array_equal(
        aligned[FEATURE + "_fit"], aligned[FEATURE + "_independent"]
    )
    plan = json.loads((directory / "resolution_plan.json").read_text())
    require(plan == resolution_plan(TAIL), "Uncertainty resolution plan changed")
    primary = json.loads((directory / "decision.json").read_text())
    final = json.loads((directory / "final_decision.json").read_text())
    require(
        final["primary_decision_sha256"] == file_sha256(directory / "decision.json"),
        "Primary decision changed",
    )
    completion = json.loads((directory / "procedure_completion.json").read_text())
    for identity in completion["artifacts"].values():
        require(
            file_sha256(Path(identity["path"])) == identity["sha256"],
            "Final artifact changed",
        )
    if not primary["accepted"]:
        require(
            final["decision"] == primary and not final["decision"]["accepted"],
            "Failed primary was rescued",
        )
        require(
            all(
                value == "not_run_primary_failed" for value in final["stages"].values()
            ),
            "Unexpected confirmation",
        )
        require(
            not (directory / "confirmation").exists(),
            "Failed primary received extra draws",
        )
    else:
        # A positive procedure needs independently audited fixed simulation draws.
        require(
            not final["decision"]["accepted"],
            "Positive procedure needs separate full-resolution audit",
        )
    result = dict(
        status="passed",
        independent_feature_values=1375,
        models_fitted=0,
        failed_primary_preserved=not primary["accepted"],
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(
        not (directory / "feature_lifecycle_verification.json").exists(),
        "Preserve old verification",
    )
    write_json(directory / "feature_lifecycle_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

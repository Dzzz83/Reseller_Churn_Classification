"""Independent direct floor, cutoff, decision and operating-count audit; no fits."""

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
CANDIDATE = "positive_floor_veto"


def direct_cutoff(reference, scores, floor):
    y = reference.churn.to_numpy() == 1
    target = int((y & reference.score.ge(floor)).sum())
    options = []
    for cut in sorted(set([floor, *scores])):
        alert = scores >= cut
        tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
        if cut >= floor and tp >= target and 10 * tp > 9 * y.sum():
            options.append((tp / (tp + fp), tp, cut))
    require(bool(options), "No feasible cutoff")
    return float(max(options)[2])


def verify(directory):
    verify_manifest(directory)
    pre = json.loads((directory / "preexecution_identity.json").read_text())
    for identity in pre["implementation"].values():
        path = directory / "source_snapshot" / Path(identity["path"]).name
        require(file_sha256(path) == identity["sha256"], "Frozen code identity changed")
    for identity in pre["inputs"].values():
        require(
            file_sha256(Path(identity["path"])) == identity["sha256"],
            "Input identity changed",
        )
    source = directory.parent / "retained-calibration-20261018"
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    folds = []
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        original, folder = source / relative, directory / relative
        verify_manifest(original)
        verify_manifest(folder)
        inner = read_csv(original / "inner_scores.csv")
        baseline = ordered(inner[inner.candidate.eq("baseline")])
        component = ordered(inner[inner.candidate.eq("retained_logistic")])
        pd.testing.assert_frame_equal(
            baseline[KEYS + ["churn"]], component[KEYS + ["churn"]]
        )
        pd.testing.assert_frame_equal(
            baseline[KEYS + ["churn"]], ordered(train[KEYS + ["churn"]])
        )
        cut = independent_cutoff(baseline.churn.to_numpy(), baseline.score.to_numpy())
        caught = baseline.churn.eq(1) & baseline.score.ge(cut)
        floor = min(component.loc[caught, "score"])
        choice = dict(
            reference_cutoff=cut,
            veto_floor=floor,
            protected_inner_cases=int(caught.sum()),
            fallback=cut == 0.0,
        )
        require(
            choice == json.loads((folder / "selection.json").read_text()),
            "Training floor changed",
        )
        transformed = baseline.score.to_numpy().copy()
        if cut != 0:
            transformed[component.score.to_numpy() < floor] = 0.0
        require(
            direct_cutoff(baseline, baseline.score.to_numpy(), cut) == cut,
            "Baseline cutoff differs",
        )
        require(direct_cutoff(baseline, transformed, cut) == cut, "Veto cutoff differs")
        protected = baseline.loc[caught, KEYS + ["churn", "score"]].copy()
        protected["component_score"] = component.loc[caught, "score"].to_numpy()
        pd.testing.assert_frame_equal(
            ordered(protected), ordered(read_csv(folder / "protected_inner_cases.csv"))
        )
        saved_inner = read_csv(folder / "inner_scores.csv")
        np.testing.assert_array_equal(
            ordered(saved_inner[saved_inner.candidate.eq(CANDIDATE)]).score, transformed
        )
        # No outer outcomes were used to reconstruct the floor above.
        outer = read_csv(original / "predictions.csv")
        base = ordered(outer[outer.strategy.eq("baseline")])
        extra = ordered(outer[outer.strategy.eq("retained_logistic")])
        expected = base.assign(strategy=CANDIDATE)
        if cut != 0:
            expected.loc[extra.score.lt(floor), "score"] = 0.0
        saved = read_csv(folder / "predictions.csv")
        pd.testing.assert_frame_equal(
            ordered(saved[saved.strategy.eq(CANDIDATE)]), expected
        )
        pd.testing.assert_frame_equal(
            ordered(saved[saved.strategy.eq("baseline")]), base
        )
        require((~expected.score.ge(cut) | base.score.ge(cut)).all(), "Added alert")
        for _, rows in saved.groupby("strategy"):
            pd.testing.assert_frame_equal(
                ordered(rows[KEYS + ["churn", "n_orders_3m"]]),
                ordered(valid[KEYS + ["churn", "n_orders_3m"]]),
            )
        for name in ["model_identities.json", "cached_hgb_identities.json"]:
            require(
                file_sha256(folder / name) == file_sha256(original / name),
                "Component identity changed",
            )
        folds.append(saved)
    predictions = read_csv(directory / "validation_predictions.csv")
    pd.testing.assert_frame_equal(
        ordered(pd.concat(folds, ignore_index=True)), ordered(predictions)
    )
    child = directory / CANDIDATE
    metrics = pd.read_csv(child / "operating_metrics.csv")
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
            "Operating counts changed",
        )
    completion = json.loads((child / "completion_manifest.json").read_text())
    for name, digest in completion["outputs"].items():
        require(file_sha256(child / name) == digest, "Primary completion changed")
    final = json.loads((child / "final_decision.json").read_text())
    primary = json.loads((child / "decision.json").read_text())
    require(
        final["primary_decision_sha256"] == file_sha256(child / "decision.json"),
        "Decision binding changed",
    )
    if not primary["accepted"]:
        require(
            final["decision"] == primary
            and set(final["stages"].values()) == {"not_run_primary_failed"},
            "Failed primary received rescue",
        )
    else:
        raise ValueError(
            "A provisional winner needs a separate full confirmation/closed-form audit"
        )
    result = dict(
        status="passed",
        training_floors=15,
        matched_predictions=len(predictions),
        operating_rows=len(metrics),
        models_fitted=0,
        alert_subset_verified=True,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(
        not (directory / "independent_verification.json").exists(), "Preserve audit"
    )
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

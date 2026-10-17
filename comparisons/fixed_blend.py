"""Evaluate one predeclared score blend from verified saved component fits."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]
CANDIDATE = "fixed_equal_blend"
COMPONENT = "cadence_no_count"


def combine_components(frame, component_id=COMPONENT, candidate_id=CANDIDATE):
    """Fixed arithmetic mean; labels/folds must match but never set the weight."""
    require(
        set(frame.strategy) == {"baseline", component_id},
        "Missing or unexpected component",
    )
    require(not frame.duplicated([*KEYS, "strategy"]).any(), "Duplicate component case")
    baseline = (
        frame[frame.strategy.eq("baseline")]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    cadence = (
        frame[frame.strategy.eq(component_id)]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    metadata = [c for c in frame.columns if c not in ("strategy", "score", "cutoff")]
    pd.testing.assert_frame_equal(baseline[metadata], cadence[metadata])
    require(
        np.isfinite(frame.score).all() and frame.score.between(0, 1).all(),
        "Invalid component score",
    )
    result = baseline.drop(columns="cutoff", errors="ignore").copy()
    result["score"] = 0.5 * baseline.score + 0.5 * cadence.score
    result["strategy"] = candidate_id
    return result


def run(source, output):
    require(not output.exists(), "Preserve existing output")
    verify_manifest(source)
    verification = json.loads((source / "independent_verification.json").read_text())
    require(
        verification["status"] == "passed",
        "Source study was not independently verified",
    )
    source_protocol = json.loads((source / "protocol.json").read_text())
    require(
        source_protocol["repeat_seeds"] == [301, 919, 1201], "Unexpected source repeats"
    )
    features = read_csv(source / "development_input.csv")
    history_path = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_path)
    history = read_csv(history_path / "case_histories.csv")
    profiles = history[[*KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol = ROOT / "comparisons/fixed_blend_protocol.md"
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    shutil.copyfile(source / "development_input.csv", out / "development_input.csv")
    write_json(
        out / "protocol.json",
        {
            "description": "Fixed equal-weight baseline/cadence blend",
            "arms": [
                source_protocol["arms"][0],
                {
                    "id": CANDIDATE,
                    "features": source_protocol["arms"][1]["features"],
                    "components": source_protocol["arms"],
                    "weights": [0.5, 0.5],
                },
            ],
            "splits": source_protocol["splits"],
            "repeat_seeds": source_protocol["repeat_seeds"],
            "interval_tail_probability": 0.05 / 34,
            "models_fitted": 0,
            "component_fits_reused": 600,
            "october_outcomes_accessed": False,
        },
    )
    predictions = []
    for split in source_protocol["splits"]:
        repeat, fold = split["repeat"], split["outer_fold"]
        train, valid = validate_partition(features, split)
        source_fold = source / f"repeat_{repeat}/fold_{fold}"
        verify_manifest(source_fold)
        folder = new_output_directory(out / f"repeat_{repeat}/fold_{fold}")
        inner = read_csv(source_fold / "inner_scores.csv").rename(
            columns={"candidate": "strategy"}
        )
        combined = combine_components(inner)
        require(
            set(combined.StoreID).isdisjoint(valid.StoreID),
            "Outer reseller entered inner selection",
        )
        baseline_inner = inner[inner.strategy.eq("baseline")]
        baseline_selection = study.select_threshold(
            baseline_inner.churn.to_numpy(), baseline_inner.score.to_numpy()
        )
        selection = study.select_threshold(
            combined.churn.to_numpy(), combined.score.to_numpy()
        )
        outer = read_csv(source_fold / "predictions.csv")
        outer = outer[outer.strategy.isin(["baseline", COMPONENT])].copy()
        baseline = outer[outer.strategy.eq("baseline")]
        require(
            baseline.cutoff.eq(baseline_selection["cutoff"]).all(),
            "Baseline cutoff differs",
        )
        blend = combine_components(outer)
        blend["cutoff"] = selection["cutoff"]
        rows = pd.concat([baseline, blend], ignore_index=True)
        predictions.append(rows)
        rows.to_csv(folder / "predictions.csv", index=False)
        pd.concat([baseline_inner, combined]).rename(
            columns={"strategy": "candidate"}
        ).to_csv(folder / "inner_scores.csv", index=False)
        pd.DataFrame(
            [
                {"candidate": "baseline", **baseline_selection},
                {"candidate": CANDIDATE, **selection},
            ]
        ).to_csv(folder / "inner_selections.csv", index=False)
        for src, dest in [
            ("inner_scores.csv", "component_inner_scores.csv"),
            ("predictions.csv", "component_outer_scores.csv"),
            ("model_identities.json", "component_model_identities.json"),
            ("split.json", "split.json"),
        ]:
            shutil.copyfile(source_fold / src, folder / dest)
        write_manifest(
            folder,
            {
                "models_fitted": 0,
                "component_fits_reused": 40,
                "source": input_identity(
                    {"source_fold_manifest": source_fold / "manifest.json"}
                ),
            },
        )
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        features, predictions, profiles, out, CANDIDATE, 0.05 / 34
    )
    write_json(
        out / "baseline_fidelity.json",
        {
            "status": "same saved baseline scores and independently reproduced inner cutoff",
            "changed_baseline_decisions": 0,
        },
    )
    write_manifest(
        out,
        {
            "stage": "fixed_blend",
            "models_fitted": 0,
            "component_fits_reused": 600,
            "decision": decision,
            "inputs": input_identity(
                {
                    "component_study_manifest": source / "manifest.json",
                    "component_verification": source / "independent_verification.json",
                    "protocol": protocol,
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "selection": Path(study.__file__),
                    "paired_analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/cadence-stability-20261016"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

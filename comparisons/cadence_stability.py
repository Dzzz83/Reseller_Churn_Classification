"""Predeclared new-partition stability check of a previously fitted configuration."""

import argparse
import json
from pathlib import Path

import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import (
    read_csv,
    require,
    validate_partition,
    verify_baseline,
)
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
REPEATS = (301, 919, 1201)
CANDIDATE = "cadence_no_count"


def run(output):
    require(not output.exists(), "Preserve existing output")
    source = ROOT / "results/repeated-grouped-precision"
    data, _, audit = verify_baseline(
        source,
        ROOT / "results/research-provenance-20261016/orders_clean_historical_bytes.csv",
    )
    histories = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(histories)
    history = read_csv(histories / "case_histories.csv")
    profiles = history[[*study.KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    grid = [
        dict(
            id="baseline",
            features=list(study.FEATURES),
            parameters=dict(study.BASE_PARAMETERS),
        ),
        dict(
            id=CANDIDATE,
            features=study.FEATURES + ["overdue_ratio", "gap_cv"],
            parameters={
                **study.BASE_PARAMETERS,
                "max_leaf_nodes": 7,
                "l2_regularization": 1.0,
            },
        ),
    ]
    splits = []
    for repeat in REPEATS:
        for fold, (ti, vi) in enumerate(study.grouped_splits(data, 5, repeat)):
            split = dict(
                repeat=repeat,
                outer_fold=fold,
                train_rows=ti.tolist(),
                validation_rows=vi.tolist(),
                train_store_ids=sorted(data.iloc[ti].StoreID.unique().tolist()),
                validation_store_ids=sorted(data.iloc[vi].StoreID.unique().tolist()),
            )
            validate_partition(data, split)
            splits.append(split)
    out = new_output_directory(output)
    protocol = ROOT / "comparisons/cadence_stability_protocol.md"
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    data.to_csv(out / "development_input.csv", index=False)
    write_json(
        out / "protocol.json",
        dict(
            arms=grid,
            splits=splits,
            repeat_seeds=list(REPEATS),
            model_seeds=list(study.MODEL_SEEDS),
            inner_seed=78,
            inner_folds=3,
            interval_tail_probability=0.05 / 28,
            historical_audit=audit,
            adaptive_selection="unused byproduct of shared evaluator",
            october_outcomes_accessed=False,
        ),
    )
    parts = []
    for split in splits:
        directory = out / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        study.evaluate_outer_fold(
            data,
            split["repeat"],
            split["outer_fold"],
            pd.Index(split["train_rows"]).to_numpy(),
            pd.Index(split["validation_rows"]).to_numpy(),
            directory,
            grid,
            3,
            study.MODEL_SEEDS,
        )
        write_manifest(directory, {"fits": 40})
        rows = read_csv(directory / "predictions.csv")
        parts.append(rows[rows.strategy.isin(["baseline", CANDIDATE])])
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        data, predictions, profiles, out, CANDIDATE, 0.05 / 28
    )
    write_json(
        out / "baseline_fidelity.json",
        {
            "status": "new outer partitions; historical scores are not direct fidelity targets",
            "historical_reference": "both previous current-runtime baselines reproduced all original decisions",
        },
    )
    write_manifest(
        out,
        {
            "stage": "cadence_stability",
            "fits": 600,
            "decision": decision,
            "inputs": input_identity(
                {
                    "protocol": protocol,
                    "history_manifest": histories / "manifest.json",
                    "source_manifest": source / "manifest.json",
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "fitting": Path(study.__file__),
                    "paired_analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

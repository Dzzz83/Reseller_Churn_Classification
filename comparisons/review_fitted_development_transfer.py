"""Saved-cutoff development transfer and actual component-size tables, no fitting."""

import argparse
import json
from pathlib import Path

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv
from comparisons.development_reporting import cutoff_transfer_tables
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)


def run(source, output):
    verify_manifest(source)
    protocol = json.loads((source / "protocol.json").read_text())
    data = read_csv(source / "development_input.csv")
    tables, sizes = [], []
    for split in protocol["splits"]:
        folder = source / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        tables.append(
            cutoff_transfer_tables(
                read_csv(folder / "inner_scores.csv"),
                read_csv(folder / "predictions.csv"),
                data,
            )
        )
        identities = json.loads((folder / "model_identities.json").read_text())
        for identity in identities:
            sizes.append(
                dict(
                    repeat=split["repeat"],
                    outer_fold=split["outer_fold"],
                    candidate=identity["candidate"],
                    stage=identity["stage"],
                    inner_fold=identity.get("inner_fold", -1),
                    seed=identity.get("seed", "deterministic"),
                    recent_purchase=identity.get("recent_purchase", "all"),
                    training_rows=identity.get("training_rows"),
                    training_positives=identity.get("training_positives"),
                    seed_score_range=identity.get("seed_score_range", 0.0),
                )
            )
    out = new_output_directory(output)
    pd.concat(tables, ignore_index=True).rename(
        columns={
            "training_rows_min": "ensemble_fitting_pool_rows_min",
            "training_rows_max": "ensemble_fitting_pool_rows_max",
            "training_resellers_min": "ensemble_fitting_pool_resellers_min",
            "training_resellers_max": "ensemble_fitting_pool_resellers_max",
        }
    ).to_csv(out / "cutoff_transfer.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "component_sizes.csv", index=False)
    (out / "report.md").write_text(
        "# Saved development operating transfer\n\n"
        "Every saved outer cutoff is applied unchanged to pooled inner OOF,each "
        "inner fold,and outer scores for both arms and activities. Pool sizes are "
        "the complete fitting population; component_sizes.csv gives actual expert "
        "subsets where recorded. Missing legacy component counts are not invented. "
        "Differences mix selection,refitting and population; no causal calibration "
        "or sample-size claim follows. No model fitting,loading or cutoff search.\n"
    )
    write_manifest(
        out,
        dict(
            inputs=input_identity(
                {"source": source / "manifest.json", "script": Path(__file__)}
            ),
            transfer_rows=sum(len(t) for t in tables),
            models_fitted=0,
        ),
    )
    print(
        f"Verified {len(tables)} folds; wrote {sum(len(t) for t in tables)} transfer rows"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

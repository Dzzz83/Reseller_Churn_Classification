"""No-fit development-only review of saved fixed numeric-feature comparisons."""

import argparse
import json
from pathlib import Path

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, validate_partition
from comparisons.breadth_reporting import markdown_table
from comparisons.development_checks import validate_development_cases
from comparisons.development_reporting import (
    cutoff_transfer_tables,
    paired_operating_tables,
)
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)


def review(source, output, *, canonical=True):
    """Read the explicit development layout, never a final-evaluation bundle."""
    manifest_path = source / "manifest.json"
    declaration = json.loads(manifest_path.read_text())
    if declaration.get("stage") != "fixed_numeric_feature":
        raise ValueError(
            "Only saved fixed_numeric_feature development studies are supported"
        )
    # Read/validate the declared development table before traversing fold evidence.
    data = read_csv(source / "development_input.csv")
    validate_development_cases(data)
    if canonical:
        from reseller_churn.config.validation_settings import EXPECTED_LABELED_COUNTS

        counts = {
            day.strftime("%Y-%m-%d"): (len(rows), int(rows.churn.sum()))
            for day, rows in data.groupby("snapshot")
        }
        if counts != EXPECTED_LABELED_COUNTS or data.StoreID.nunique() != 422:
            raise ValueError(
                "Development population differs from the approved canonical cases"
            )
    verify_manifest(source)
    protocol = json.loads((source / "protocol.json").read_text())
    candidate = protocol["arms"][1]["id"]
    for arm in protocol["arms"]:
        validate_development_cases(data, arm["features"])
    predictions = read_csv(source / "validation_predictions.csv")
    metrics, paired, worst = paired_operating_tables(predictions, candidate)
    reference = (
        data[["StoreID", "snapshot", "churn", "n_orders_3m"]]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    for _, part in predictions.groupby(["repeat", "strategy"]):
        pd.testing.assert_frame_equal(
            part[reference.columns]
            .sort_values(["snapshot", "StoreID"])
            .reset_index(drop=True),
            reference,
        )
    actual_splits = set(
        map(tuple, predictions[["repeat", "outer_fold"]].drop_duplicates().to_numpy())
    )
    declared_splits = [
        (split["repeat"], split["outer_fold"]) for split in protocol["splits"]
    ]
    if (
        len(declared_splits) != len(set(declared_splits))
        or set(declared_splits) != actual_splits
    ):
        raise ValueError("Declared splits do not match saved validation predictions")
    transfers, split_audits = [], []
    inputs = dict(source_manifest=manifest_path)
    for split in protocol["splits"]:
        folder = source / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        expected = predictions[
            predictions.repeat.eq(split["repeat"])
            & predictions.outer_fold.eq(split["outer_fold"])
        ]
        sort = ["strategy", "snapshot", "StoreID"]
        pd.testing.assert_frame_equal(
            outer.sort_values(sort).reset_index(drop=True),
            expected.sort_values(sort).reset_index(drop=True),
        )
        transfers.append(cutoff_transfer_tables(inner, outer, data))
        train, valid = validate_partition(data, split)
        for _, part in inner.groupby("candidate"):
            pd.testing.assert_frame_equal(
                part[["StoreID", "snapshot", "churn"]]
                .sort_values(["snapshot", "StoreID"])
                .reset_index(drop=True),
                train[["StoreID", "snapshot", "churn"]]
                .sort_values(["snapshot", "StoreID"])
                .reset_index(drop=True),
            )
        for day, rows in valid.groupby("snapshot"):
            unresolved = (train.snapshot + pd.DateOffset(months=6)).gt(day)
            split_audits.append(
                dict(
                    repeat=split["repeat"],
                    outer_fold=split["outer_fold"],
                    snapshot=day.strftime("%Y-%m-%d"),
                    validation_rows=len(rows),
                    training_rows=len(train),
                    training_resellers=train.StoreID.nunique(),
                    validation_resellers=rows.StoreID.nunique(),
                    shared_resellers=0,
                    training_labels_unresolved_at_validation=int(unresolved.sum()),
                    design="grouped; dates mixed, not a forecast simulation",
                )
            )
        inputs[str(folder.relative_to(source) / "manifest.json")] = (
            folder / "manifest.json"
        )
    out = new_output_directory(output)
    metrics.to_csv(out / "operating_metrics.csv", index=False)
    paired.to_csv(out / "paired_operating_metrics.csv", index=False)
    worst.to_csv(out / "worst_slices.csv", index=False)
    pd.concat(transfers, ignore_index=True).to_csv(
        out / "cutoff_transfer.csv", index=False
    )
    pd.DataFrame(split_audits).to_csv(out / "split_timing.csv", index=False)
    write_json(
        out / "review.json",
        dict(
            candidate=candidate,
            models_fitted=0,
            cutoffs_selected=0,
            test_access=False,
            historical_acceptance_changed=False,
            interpretation="Harm flags and worst slices are descriptive, not new acceptance criteria.",
            transfer_limitations="Different models and populations; not causal attribution or calibration proof.",
            validation_views="Grouped and temporal answer different questions; temporal selection is not assumed superior.",
        ),
    )
    periods = paired[paired.scope.eq("repeat_pooled") & paired.slice.eq("snapshot")]
    overall = paired[paired.scope.eq("repeat_pooled") & paired.slice.eq("overall")]
    columns = [
        "repeat",
        "group",
        "rows_baseline",
        "churners_baseline",
        "recall_baseline",
        "recall_candidate",
        "fp_baseline",
        "fp_candidate",
        "false_alert_reduction",
        "recall_difference",
    ]
    report = "# Development-only matched review\n\n"
    report += "No models fitted, cutoffs selected, test accessed, or historical acceptance changed.\n\n"
    report += "## Every repeat\n\n" + markdown_table(overall[columns]) + "\n\n"
    report += (
        "## Every snapshot and repeat\n\n" + markdown_table(periods[columns]) + "\n\n"
    )
    report += "All fold/activity/intersection counts and undefined-rate flags are in operating_metrics.csv and paired_operating_metrics.csv. Worst slices are sorted in worst_slices.csv; harm flags are descriptive, not acceptance decisions.\n\n"
    report += "cutoff_transfer.csv applies each saved outer cutoff unchanged to inner OOF, individual inner-fold and outer predictions, with actual component training-size ranges. Differences combine refitting, selection and population effects; no calibration or causal conclusion follows. The pooled-OOF row describes several models, not one training size.\n\n"
    report += "split_timing.csv labels grouped folds as mixed-period evidence. It does not make temporal selection a proven remedy. Repeated use of development data is legitimate model development; confidence must reflect the evidence available. No newly untouched-data claim is made.\n"
    (out / "report.md").write_text(report)
    write_manifest(
        out,
        dict(
            stage="development_only_review",
            inputs=input_identity(inputs),
            implementation=input_identity(
                {
                    "review": Path(__file__),
                    "reporting": Path(__file__).with_name("development_reporting.py"),
                    "guards": Path(__file__).with_name("development_checks.py"),
                }
            ),
        ),
    )
    print(f"Saved development-only review to {out}; no fits or cutoff selection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.source, args.output)

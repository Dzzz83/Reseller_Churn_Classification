"""Matched historical descriptors for the fixed cadence procedure's residuals."""

import argparse
from pathlib import Path

import numpy as np

from comparisons.baseline_diagnosis.history_comparison import (
    DESCRIPTORS,
    match_cases,
    matched_differences,
    summarize_differences,
)
from comparisons.baseline_diagnosis.inputs import read_csv, require
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    histories = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(histories)
    data = read_csv(source / "development_input.csv")
    history = read_csv(histories / "case_histories.csv")
    predictions = read_csv(source / "validation_predictions.csv")
    predictions = predictions[predictions.strategy.eq("cadence_no_count")].copy()
    require(
        len(predictions) == 4125
        and not predictions.duplicated(KEYS + ["repeat"]).any(),
        "Incomplete repeated predictions",
    )
    predictions["alert"] = predictions.score.ge(predictions.cutoff)
    alerts = predictions.groupby(KEYS).alert.sum().rename("alerts").reset_index()
    cases = (
        data[KEYS + ["churn", *FEATURES]]
        .merge(history[KEYS + DESCRIPTORS], on=KEYS, validate="one_to_one")
        .merge(alerts, on=KEYS, validate="one_to_one")
    )
    for feature in ["gap_cv", "overdue_ratio"]:
        check = cases.merge(
            data[KEYS + [feature]],
            on=KEYS,
            validate="one_to_one",
            suffixes=("", "_saved"),
        )
        np.testing.assert_allclose(
            check[feature], check[f"{feature}_saved"], equal_nan=True, rtol=1e-12
        )
    matches, scales, coverage = match_cases(cases)
    differences = matched_differences(cases, matches)
    summary = summarize_differences(differences)
    out = new_output_directory(output)
    for name, table in [
        ("cases", cases),
        ("matches", matches),
        ("matching_scales", scales),
        ("coverage", coverage),
        ("matched_differences", differences),
        ("summary", summary),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    write_manifest(
        out,
        dict(
            stage="cadence_residual_history",
            models_fitted=0,
            cutoffs_selected=0,
            october_outcomes_accessed=False,
            inputs=input_identity(
                dict(
                    source=source / "manifest.json",
                    histories=histories / "manifest.json",
                    protocol=ROOT / "comparisons/persistent_error_investigation.md",
                )
            ),
            implementation=input_identity(
                dict(
                    runner=Path(__file__),
                    matching=ROOT
                    / "comparisons/baseline_diagnosis/history_comparison.py",
                )
            ),
        ),
    )
    print(
        coverage[coverage.k.eq(3)]
        .groupby("comparison")
        .agg(targets=("StoreID", "size"), minimum_controls=("matched_controls", "min"))
        .to_string()
    )
    print(
        summary[summary.k.eq(3) & summary.feature.eq("last_order_unit_revenue")][
            [
                "comparison",
                "scope",
                "snapshot",
                "targets",
                "mean_difference",
                "median_difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

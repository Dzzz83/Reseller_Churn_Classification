"""Fixed historical matching audit of four quarter-count profiles, without fitting."""

import argparse
from pathlib import Path

from comparisons.baseline_diagnosis.history_comparison import (
    matched_differences,
    summarize_differences,
)
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.breadth_reporting import markdown_table
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def run(output, *, residual=False):
    source = ROOT / "results/recency-shape-diagnosis-20261018"
    matching = (
        ROOT
        / "results"
        / (
            "cadence-residual-history-20261017"
            if residual
            else "persistent-error-investigation-20261016"
        )
    )
    verify_manifest(source)
    verify_manifest(matching)
    cases = read_csv(source / "quarter_counts.csv")
    require(cases.n_orders_12m.gt(0).all(), "Annual count must be positive")
    descriptors = [f"orders_quarter_{i}" for i in range(1, 5)]
    for quarter in range(1, 5):
        name = f"share_quarter_{quarter}"
        cases[name] = cases[f"orders_quarter_{quarter}"] / cases.n_orders_12m
        descriptors.append(name)
    matches = read_csv(matching / "matches.csv")
    differences = matched_differences(cases, matches, descriptors)
    summary = summarize_differences(differences)
    out = new_output_directory(output)
    cases.to_csv(out / "case_profiles.csv", index=False)
    differences.to_csv(out / "matched_differences.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    recent = summary[
        summary.scope.eq("snapshot")
        & summary.comparison.eq("recent_false_alert_vs_caught")
    ]
    (out / "report.md").write_text(
        "# Historical-quarter profile diagnosis\n\n"
        "All four pre-snapshot order-count quarters and all four annual-normalized shares are shown. "
        "The original baseline-space neighbors, comparison groups, and k=1/3/5 choices are unchanged. "
        "No estimator, selector, cutoff or candidate is fitted. Differences are target minus control. "
        "For recent_false_alert_vs_caught, targets are persistent continuing-account false alerts and "
        "controls are caught churners. Matching is descriptive and does not identify causal effects.\n\n"
        + markdown_table(
            recent[
                [
                    "feature",
                    "k",
                    "snapshot",
                    "targets",
                    "median_difference",
                    "mean_difference",
                ]
            ]
        )
        + "\n\nComplete inactive and missed-customer comparisons are retained in summary.csv. "
        "Single-year quarter profiles do not establish repeatable seasonality, and repeated customers "
        "and shared controls are dependent. No significance or independent confirmation is claimed. "
        "No favorable quarter or representation is automatically selected.\n"
    )
    write_manifest(
        out,
        dict(
            models_fitted=0,
            descriptors=descriptors,
            inputs=input_identity(
                {
                    "profiles": source / "manifest.json",
                    "matches": matching / "manifest.json",
                    "script": Path(__file__),
                }
            ),
        ),
    )
    print(
        recent[recent.feature.str.startswith("share")][
            ["feature", "k", "snapshot", "mean_difference"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()
    run(args.output, residual=args.residual)

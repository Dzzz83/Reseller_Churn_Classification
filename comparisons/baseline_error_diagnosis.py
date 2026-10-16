"""Describe saved held-out baseline errors without fitting or selecting models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.analysis import (
    build_cases,
    historical_profiles,
    metric_tables,
    ranking_tables,
    stability_table,
    verify_saved_metrics,
)
from comparisons.baseline_diagnosis.definitions import (
    POLICIES,
    RANK_BANDS,
    REPEATS,
    ROOT,
    SLICES,
)
from comparisons.baseline_diagnosis.inputs import require, verify_baseline
from comparisons.baseline_diagnosis.reporting import plot_diagnostics, write_report
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    write_json,
    write_manifest,
)


def run(source: Path, orders_path: Path, output: Path) -> Path:
    require(not output.exists(), "Output directory already exists; choose a new run ID")
    features, predictions, audit = verify_baseline(source, orders_path)
    implementation = [
        Path(__file__),
        *sorted((Path(__file__).parent / "baseline_diagnosis").glob("*.py")),
    ]
    audit["implementation"] = input_identity(
        {path.name: path for path in implementation}
    )
    # Only historical source facts are retained; no future order aggregation or label construction.
    orders = pd.read_csv(
        orders_path,
        usecols=["StoreID", "SalesOrderID", "OrderDate"],
        parse_dates=["OrderDate"],
    )
    orders = orders[orders.OrderDate.lt(features.snapshot.max())]
    profiles = historical_profiles(features, orders)
    np.testing.assert_array_equal(
        profiles.sort_values(["snapshot", "StoreID"]).n_orders_12m,
        features.n_orders_12m,
    )
    cases = build_cases(features, predictions, profiles)
    cases["high_rank_false_alert"] = cases.result_type.eq(
        "false_alert"
    ) & cases.rank_percentile.le(0.25)
    cases["low_rank_missed_churn"] = cases.result_type.eq(
        "missed_churn"
    ) & cases.rank_percentile.ge(0.75)
    metrics, bands = metric_tables(cases)
    curves, burden = ranking_tables(cases)
    stability = stability_table(cases)
    reproduced = verify_saved_metrics(metrics, source)
    output = new_output_directory(output)
    protocol = {
        "purpose": "descriptive development-only baseline diagnosis",
        "repeats": list(REPEATS),
        "policies": POLICIES,
        "slices": SLICES,
        "rank_bands": RANK_BANDS,
        "rank_definition": "(descending midrank - 0.5) / outer validation fold size",
        "near_boundary_percentile_distance": 0.05,
        "unstable_rank_range": 0.20,
        "high_recall_targets": [0.90, 0.95, 1.0],
        "high_recall_points_are_diagnostic_only": True,
        "small_sample": "rows <30 or churners <10 or non-churners <10",
        "uncertainty": "descriptive counts only; no independence or interval claims",
        "future_acceptance": "match or exceed baseline recall on same splits with fewer false alerts",
        "models_fitted": 0,
        "cutoffs_selected": 0,
        "final_outcomes_accessed": False,
    }
    write_json(output / "protocol.json", protocol)
    for name, table in {
        "account_diagnostics": cases,
        "subgroup_metrics": metrics,
        "rank_band_metrics": bands,
        "ranking_curves": curves,
        "high_recall_burden": burden,
        "account_stability": stability,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    verification = {
        "status": "passed",
        "saved_metric_rows_reproduced": reproduced,
        "development_cases": len(features),
        "resellers": features.StoreID.nunique(),
        "case_policy_repeat_rows": len(cases),
        "outer_splits_verified": 15,
        "models_fitted": 0,
        "cutoffs_selected": 0,
        "final_outcomes_accessed": False,
    }
    write_json(output / "verification.json", verification)
    plot_diagnostics(cases, curves, metrics, output)
    write_report(cases, metrics, burden, stability, verification, output)
    write_manifest(
        output, {"stage": "baseline_error_diagnosis", "audit": audit, **verification}
    )
    print(f"Saved baseline diagnosis: {output}")
    print(json.dumps(verification, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/repeated-grouped-precision"
    )
    parser.add_argument(
        "--orders", type=Path, default=ROOT / "datasets/orders_clean.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.orders, args.output)


if __name__ == "__main__":
    main()

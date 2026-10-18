"""Shared source-file preparation for CLI and grouped development scripts."""

from pathlib import Path

import pandas as pd

from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    write_manifest,
)
from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS
from reseller_churn.data.preparation import (
    build_development,
    build_features,
    date,
    load_sources,
)


def add_source_arguments(parser):
    for name in ("orders", "stores", "headers", "lines"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--coverage-from", required=True)
    parser.add_argument(
        "--coverage-through",
        required=True,
        help="Exclusive end of complete source coverage.",
    )
    parser.add_argument(
        "--approved-excluded-order-ids", type=int, nargs="*", default=[]
    )


def write_preparation(
    paths,
    coverage_from,
    coverage_through,
    output,
    *,
    snapshot=None,
    approved_excluded_order_ids=(),
) -> Path:
    stage = "prepare" if snapshot is None else "features"
    needed_through = (
        date(TRAINING_SNAPSHOTS[-1]) + pd.DateOffset(months=6)
        if snapshot is None
        else date(snapshot)
    )
    source = load_sources(
        paths,
        coverage_from,
        coverage_through,
        needed_through,
        approved_excluded_order_ids=approved_excluded_order_ids,
    )
    data = (
        build_development(source)
        if snapshot is None
        else build_features(source, snapshot)
    )
    output = new_output_directory(output)
    name = "development.csv" if snapshot is None else "features.csv"
    data.to_csv(output / name, index=False)
    write_manifest(
        output,
        {
            "stage": stage,
            "inputs": input_identity(paths),
            "coverage_from": str(date(coverage_from).date()),
            "coverage_through": str(date(coverage_through).date()),
            "loaded_through": str(source.loaded_through.date()),
            "approved_excluded_order_ids": list(approved_excluded_order_ids),
            "rows": len(data),
            "snapshots": sorted(data.snapshot.dt.strftime("%Y-%m-%d").unique()),
            "final_outcomes_constructed": False,
            "source_audit": {
                "accepted_orders": len(source.orders),
                "accepted_lines": len(source.lines),
                "excluded_raw_order_ids": list(source.excluded_order_ids),
                "reconciliation": "passed",
            },
        },
    )
    return output

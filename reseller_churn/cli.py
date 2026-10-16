"""Explicit preparation, fixed training, ranked scoring, and cutoff evaluation."""

import argparse
from collections.abc import Sequence
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
from reseller_churn.evaluation.cutoff_evaluation import evaluate
from reseller_churn.model import predict, train


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["prepare", "features"]:
        command = commands.add_parser(
            name,
            help="Prepare development rows"
            if name == "prepare"
            else "Build features for an unlabeled scoring snapshot",
        )
        for source in ["orders", "stores", "headers", "lines"]:
            command.add_argument(f"--{source}", type=Path, required=True)
        command.add_argument(
            "--coverage-from",
            required=True,
            help="Start of complete order-source coverage",
        )
        command.add_argument(
            "--coverage-through",
            required=True,
            help="Exclusive end of complete order-source coverage",
        )
        if name == "features":
            command.add_argument("--snapshot", required=True)
        command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("train", help="Train the fixed five-seed ensemble")
    command.add_argument("--development", type=Path, required=True)
    command.add_argument("--prediction-date", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser(
        "predict", help="Save ranked scores without reading outcomes"
    )
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser(
        "evaluate", help="Compare supplied score cutoffs without training"
    )
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--outcomes", type=Path, required=True)
    command.add_argument("--cutoffs", type=float, nargs="+", required=True)
    command.add_argument(
        "--outcomes-through",
        required=True,
        help="Exclusive end of complete observed outcome coverage",
    )
    command.add_argument(
        "--cohort-status", choices=["inspected", "independent"], required=True
    )
    command.add_argument(
        "--approve-independent-evaluation",
        action="store_true",
        help="Explicit approval of an externally frozen independent evaluation",
    )
    command.add_argument("--output", type=Path, required=True)
    return root


def _run_data_command(arguments: argparse.Namespace) -> None:
    paths = {
        name: getattr(arguments, name)
        for name in ["orders", "stores", "headers", "lines"]
    }
    needed_through = (
        date(TRAINING_SNAPSHOTS[-1]) + pd.DateOffset(months=6)
        if arguments.command == "prepare"
        else date(arguments.snapshot)
    )
    source = load_sources(
        paths, arguments.coverage_from, arguments.coverage_through, needed_through
    )
    data = (
        build_development(source)
        if arguments.command == "prepare"
        else build_features(source, arguments.snapshot)
    )
    output = new_output_directory(arguments.output)
    name = "development.csv" if arguments.command == "prepare" else "features.csv"
    data.to_csv(output / name, index=False)
    write_manifest(
        output,
        {
            "stage": arguments.command,
            "inputs": input_identity(paths),
            "coverage_from": arguments.coverage_from,
            "coverage_through": arguments.coverage_through,
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


def _run_evaluation_command(arguments: argparse.Namespace) -> None:
    evaluate(
        arguments.predictions,
        arguments.outcomes,
        arguments.output,
        arguments.cutoffs,
        arguments.outcomes_through,
        arguments.cohort_status,
        arguments.approve_independent_evaluation,
    )
    metrics = pd.read_csv(arguments.output / "cutoff_metrics.csv")
    columns = [
        "snapshot",
        "cutoff",
        "recall",
        "precision",
        "f1",
        "tp",
        "fp",
        "fn",
        "alert_count",
    ]
    print(metrics.loc[metrics.subgroup.eq("overall"), columns].to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    if arguments.command in {"prepare", "features"}:
        _run_data_command(arguments)
    elif arguments.command == "train":
        train(arguments.development, arguments.output, arguments.prediction_date)
    elif arguments.command == "predict":
        predict(arguments.model, arguments.features, arguments.output)
    else:
        _run_evaluation_command(arguments)
    print(f"Saved {arguments.command} outputs: {arguments.output}")


if __name__ == "__main__":
    main()

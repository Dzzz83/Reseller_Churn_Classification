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
from reseller_churn.config import TRAINING_SNAPSHOTS
from reseller_churn.data import (
    build_development,
    load_sources,
)
from reseller_churn.dates import parse_calendar_date
from reseller_churn.evaluation import evaluate
from reseller_churn.features import build_features
from reseller_churn.model import predict, train


def build_command_parser() -> argparse.ArgumentParser:
    """Define the five supported operations and their required arguments."""
    command_parser = argparse.ArgumentParser(description=__doc__)
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    for command_name in ["prepare", "features"]:
        if command_name == "prepare":
            help_text = "Prepare development rows"
        else:
            help_text = "Build features for an unlabeled scoring snapshot"
        subcommand_parser = subcommands.add_parser(command_name, help=help_text)
        for source_name in ["orders", "stores", "headers", "lines"]:
            subcommand_parser.add_argument(f"--{source_name}", type=Path, required=True)
        subcommand_parser.add_argument(
            "--coverage-from",
            required=True,
            help="Start of complete order-source coverage",
        )
        subcommand_parser.add_argument(
            "--coverage-through",
            required=True,
            help="Exclusive end of complete order-source coverage",
        )
        if command_name == "features":
            subcommand_parser.add_argument("--snapshot", required=True)
        subcommand_parser.add_argument("--output", type=Path, required=True)
    subcommand_parser = subcommands.add_parser(
        "train", help="Train the fixed five-seed ensemble"
    )
    subcommand_parser.add_argument("--development", type=Path, required=True)
    subcommand_parser.add_argument("--prediction-date", required=True)
    subcommand_parser.add_argument("--output", type=Path, required=True)
    subcommand_parser = subcommands.add_parser(
        "predict", help="Save ranked scores without reading outcomes"
    )
    subcommand_parser.add_argument("--model", type=Path, required=True)
    subcommand_parser.add_argument("--features", type=Path, required=True)
    subcommand_parser.add_argument("--output", type=Path, required=True)
    subcommand_parser = subcommands.add_parser(
        "evaluate", help="Compare supplied score cutoffs without training"
    )
    subcommand_parser.add_argument("--predictions", type=Path, required=True)
    subcommand_parser.add_argument("--outcomes", type=Path, required=True)
    subcommand_parser.add_argument("--cutoffs", type=float, nargs="+", required=True)
    subcommand_parser.add_argument(
        "--outcomes-through",
        required=True,
        help="Exclusive end of complete observed outcome coverage",
    )
    subcommand_parser.add_argument(
        "--cohort-status", choices=["inspected", "independent"], required=True
    )
    subcommand_parser.add_argument(
        "--approve-independent-evaluation",
        action="store_true",
        help="Explicit approval of an externally frozen independent evaluation",
    )
    subcommand_parser.add_argument("--output", type=Path, required=True)
    return command_parser


def _run_data_command(arguments: argparse.Namespace) -> None:
    """Prepare training samples or scoring features and record the source audit."""
    source_paths = {
        "orders": arguments.orders,
        "stores": arguments.stores,
        "headers": arguments.headers,
        "lines": arguments.lines,
    }
    # Orders on or after end_date are excluded.
    if arguments.command == "prepare":
        latest_training_snapshot = parse_calendar_date(TRAINING_SNAPSHOTS[-1])
        end_date = latest_training_snapshot + pd.DateOffset(months=6)
    else:
        end_date = parse_calendar_date(arguments.snapshot)

    sources = load_sources(
        source_paths,
        arguments.coverage_from,
        arguments.coverage_through,
        end_date,
    )
    if arguments.command == "prepare":
        prepared_data = build_development(sources)
        output_filename = "development.csv"
    else:
        prepared_data = build_features(sources, arguments.snapshot)
        output_filename = "features.csv"

    output_directory = new_output_directory(arguments.output)
    prepared_data.to_csv(output_directory / output_filename, index=False)
    write_manifest(
        output_directory,
        {
            "stage": arguments.command,
            "inputs": input_identity(source_paths),
            "coverage_from": arguments.coverage_from,
            "coverage_through": arguments.coverage_through,
            "rows": len(prepared_data),
            "snapshots": sorted(
                prepared_data.snapshot.dt.strftime("%Y-%m-%d").unique()
            ),
            "final_outcomes_constructed": False,
            "source_audit": {
                "accepted_orders": len(sources.orders),
                "accepted_lines": len(sources.lines),
                "excluded_raw_order_ids": list(sources.excluded_order_ids),
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
    command_parser = build_command_parser()
    arguments = command_parser.parse_args(argv)
    if arguments.command in {"prepare", "features"}:
        _run_data_command(arguments)
    elif arguments.command == "train":
        train(arguments.development, arguments.output, arguments.prediction_date)
    elif arguments.command == "predict":
        predict(arguments.model, arguments.features, arguments.output)
    elif arguments.command == "evaluate":
        _run_evaluation_command(arguments)
    print(f"Saved {arguments.command} outputs: {arguments.output}")


if __name__ == "__main__":
    main()

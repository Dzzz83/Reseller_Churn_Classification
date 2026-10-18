"""Prepare historical inputs, develop grouped models, and export saved-model scores."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from reseller_churn.data.commands import add_source_arguments, write_preparation
from reseller_churn.evaluation.cutoff_evaluation import evaluate
from reseller_churn.model import predict, train
from reseller_churn.modeling.grouped_command import (
    add_development_arguments,
    development_options,
    run_development_file,
)
from reseller_churn.modeling.grouped_validation import GroupedValidationPlan
from reseller_churn.prediction import export_predictions


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "develop", help="Run repeated nested grouped development"
    )
    command.add_argument("--development", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--models", type=Path, required=True)
    add_development_arguments(command)
    command = commands.add_parser(
        "export", help="Export saved-model scores and flags without outcomes"
    )
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--threshold", type=float, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ["prepare", "features"]:
        command = commands.add_parser(
            name,
            help="Prepare development rows"
            if name == "prepare"
            else "Build features for an unlabeled scoring snapshot",
        )
        add_source_arguments(command)
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
        paths = {
            name: getattr(arguments, name)
            for name in ("orders", "stores", "headers", "lines")
        }
        write_preparation(
            paths,
            arguments.coverage_from,
            arguments.coverage_through,
            arguments.output,
            snapshot=getattr(arguments, "snapshot", None),
            approved_excluded_order_ids=arguments.approved_excluded_order_ids,
        )
    elif arguments.command == "train":
        train(arguments.development, arguments.output, arguments.prediction_date)
    elif arguments.command == "predict":
        predict(arguments.model, arguments.features, arguments.output)
    elif arguments.command == "export":
        export_predictions(
            arguments.model, arguments.features, arguments.output, arguments.threshold
        )
    elif arguments.command == "develop":
        options = development_options(arguments)
        plan = GroupedValidationPlan(
            options.pop("outer_folds"),
            options.pop("inner_folds"),
            tuple(options.pop("repeat_seeds")),
            options.pop("inner_seed"),
        )
        run_development_file(
            arguments.development,
            arguments.output,
            arguments.models,
            plan=plan,
            **options,
        )
    else:
        _run_evaluation_command(arguments)
    print(f"Saved {arguments.command} outputs: {arguments.output}")


if __name__ == "__main__":
    main()

"""Explicit production commands; source paths and coverage are always supplied."""

import argparse
from pathlib import Path

from reseller_churn.modeling.bundles import load_model
from reseller_churn.modeling.training import develop_from_sources
from reseller_churn.prediction import export_predictions


def source_arguments(parser):
    for name in ("orders", "stores", "headers", "lines"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--coverage-from", required=True)
    parser.add_argument("--coverage-through", required=True)
    parser.add_argument("--output", type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ten-feature period/class-weighted reseller churn"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="Export one outcome-free Power BI snapshot"
    )
    source_arguments(export)
    export.add_argument("--snapshot", required=True)
    export.add_argument("--model", type=Path, required=True)
    export.add_argument(
        "--model-manifest", type=Path, help="Defaults to packaged retained manifest"
    )
    develop = commands.add_parser(
        "develop",
        help="Future updates: nested grouped validation and five-seed training",
    )
    source_arguments(develop)
    validate = commands.add_parser(
        "validate-model", help="Validate artifact identity and runtime without scoring"
    )
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--model-manifest", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "validate-model":
        ensemble = load_model(args["model"], args["model_manifest"])
        print(f"Validated {len(ensemble.models)} estimators; no fitting or scoring.")
        return
    paths = {name: args.pop(name) for name in ("orders", "stores", "headers", "lines")}
    if command == "export":
        args["model_path"] = args.pop("model")
        result = export_predictions(paths=paths, **args)
    else:
        result = develop_from_sources(paths=paths, **args)
    print(result)

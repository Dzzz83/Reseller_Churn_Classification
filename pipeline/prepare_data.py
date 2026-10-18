"""Prepare strict ten-feature development rows from reconciled source files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reseller_churn.config.project_paths import create_run_id, get_run_directory
from reseller_churn.data.commands import add_source_arguments, write_preparation


def prepare_data(run_id, **source_options):
    output = get_run_directory(run_id) / "preparation"
    paths = {
        name: source_options.pop(name)
        for name in ("orders", "stores", "headers", "lines")
    }
    write_preparation(paths, output=output, **source_options)
    return output / "development.csv"


def source_options(arguments):
    names = (
        "orders",
        "stores",
        "headers",
        "lines",
        "coverage_from",
        "coverage_through",
        "approved_excluded_order_ids",
    )
    return {name: getattr(arguments, name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    add_source_arguments(parser)
    arguments = parser.parse_args()
    prepare_data(arguments.run_id or create_run_id(), **source_options(arguments))


if __name__ == "__main__":
    main()

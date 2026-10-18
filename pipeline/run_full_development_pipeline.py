"""Prepare reference data and run repeated nested grouped development."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.development_workflow import run_development
from pipeline.prepare_data import add_source_arguments, prepare_data, source_options
from reseller_churn.config.project_paths import create_run_id
from reseller_churn.modeling.grouped_command import (
    add_development_arguments,
    development_options,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    add_source_arguments(parser)
    add_development_arguments(parser)
    arguments = parser.parse_args()
    run_id = arguments.run_id or create_run_id()
    prepare_data(run_id, **source_options(arguments))
    run_development(run_id, **development_options(arguments))
    print(f"Completed grouped development: {run_id}")


if __name__ == "__main__":
    main()

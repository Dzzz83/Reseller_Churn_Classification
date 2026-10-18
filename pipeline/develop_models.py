"""Run repeated nested grouped development from prepared rows."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.development_workflow import run_development
from reseller_churn.modeling.grouped_command import (
    add_development_arguments,
    development_options,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    add_development_arguments(parser)
    arguments = parser.parse_args()
    run_development(arguments.run_id, **development_options(arguments))


if __name__ == "__main__":
    main()

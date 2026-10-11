from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.prepare_data import prepare_data
from pipeline.develop_models import run_development
from reseller_churn.config.project_paths import create_run_id
from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.modeling.candidates import (
    ARCHITECTURES,
    IMBALANCE_STRATEGIES,
)
from reseller_churn.evaluation.selection import DEFAULT_PRECISION_FLOORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--precision-floors",
        type=float,
        nargs="+",
        default=DEFAULT_PRECISION_FLOORS,
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES)
    parser.add_argument("--feature-sets", nargs="+", choices=tuple(FEATURE_SETS))
    parser.add_argument(
        "--imbalance-strategies",
        nargs="+",
        choices=IMBALANCE_STRATEGIES,
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    arguments = parser.parse_args()

    run_id = arguments.run_id or create_run_id()
    print("=== Prepare Data ===", flush=True)
    prepare_data(run_id)
    print("=== Develop Models ===", flush=True)
    run_development(
        run_id,
        precision_floors=tuple(arguments.precision_floors),
        tuned=arguments.tune,
        architectures=(
            tuple(arguments.architectures) if arguments.architectures else None
        ),
        feature_sets=(
            tuple(arguments.feature_sets) if arguments.feature_sets else None
        ),
        imbalance_strategies=(
            tuple(arguments.imbalance_strategies)
            if arguments.imbalance_strategies
            else None
        ),
        seeds=tuple(arguments.seeds) if arguments.seeds else None,
    )
    print("=== Development Pipeline Completed ===")
    print(f"Run ID: {run_id}")
    print("Protected final-test labels were NOT used.")


if __name__ == "__main__":
    main()

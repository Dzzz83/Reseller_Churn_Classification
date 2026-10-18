"""Supported repeated nested grouped development, using prepared development rows."""

from reseller_churn.config.project_paths import MODELS_DIR, get_run_directory
from reseller_churn.evaluation.selection import DEFAULT_PRECISION_FLOORS
from reseller_churn.modeling.grouped_command import run_development_file
from reseller_churn.modeling.grouped_validation import GroupedValidationPlan


def run_development(
    run_id,
    *,
    precision_floors=DEFAULT_PRECISION_FLOORS,
    tuned=False,
    architectures=None,
    feature_sets=None,
    imbalance_strategies=None,
    seeds=None,
    outer_folds=5,
    inner_folds=3,
    repeat_seeds=(42, 78, 88),
    inner_seed=78,
    cutoff_objective="recall_at_precision",
    minimum_recall=0.9,
):
    run_directory = get_run_directory(run_id)
    return run_development_file(
        run_directory / "preparation/development.csv",
        run_directory / "development",
        MODELS_DIR / "provisional" / run_id,
        precision_floors=precision_floors,
        tuned=tuned,
        architectures=architectures,
        feature_sets=feature_sets,
        imbalance_strategies=imbalance_strategies,
        seeds=seeds,
        plan=GroupedValidationPlan(
            outer_folds, inner_folds, tuple(repeat_seeds), inner_seed
        ),
        cutoff_objective=cutoff_objective,
        minimum_recall=minimum_recall,
    )

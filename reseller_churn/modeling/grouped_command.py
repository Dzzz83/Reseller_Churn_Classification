"""Shared arguments and file boundary for supported grouped-development commands."""

import pandas as pd

from reseller_churn.artifacts import input_identity
from reseller_churn.config.feature_sets import MODEL_FEATURE_SETS
from reseller_churn.evaluation.selection import DEFAULT_PRECISION_FLOORS
from reseller_churn.modeling.candidates import (
    ARCHITECTURES,
    IMBALANCE_STRATEGIES,
    build_candidates,
)
from reseller_churn.modeling.grouped_development import run_grouped_study
from reseller_churn.modeling.grouped_validation import (
    ThresholdPolicy,
)


def add_development_arguments(parser):
    parser.add_argument(
        "--precision-floors",
        type=float,
        nargs="+",
        default=DEFAULT_PRECISION_FLOORS,
        help="Inner pooled OOF minimum precision fractions; default 0.1 through 0.9.",
    )
    parser.add_argument(
        "--cutoff-objective",
        choices=["recall_at_precision", "precision_at_recall"],
        default="recall_at_precision",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.9,
        help="Strict recall limit for precision_at_recall only.",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES)
    parser.add_argument("--feature-sets", nargs="+", choices=tuple(MODEL_FEATURE_SETS))
    parser.add_argument(
        "--imbalance-strategies", nargs="+", choices=IMBALANCE_STRATEGIES
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[42, 78, 88])
    parser.add_argument("--inner-seed", type=int, default=78)


def development_options(arguments):
    names = (
        "precision_floors",
        "tuned",
        "architectures",
        "feature_sets",
        "imbalance_strategies",
        "seeds",
        "outer_folds",
        "inner_folds",
        "repeat_seeds",
        "inner_seed",
        "cutoff_objective",
        "minimum_recall",
    )
    return {
        name: getattr(arguments, "tune" if name == "tuned" else name) for name in names
    }


def run_development_file(
    path,
    output,
    models,
    *,
    precision_floors=DEFAULT_PRECISION_FLOORS,
    tuned=False,
    architectures=None,
    feature_sets=None,
    imbalance_strategies=None,
    seeds=None,
    plan=None,
    cutoff_objective="recall_at_precision",
    minimum_recall=0.9,
):
    if not precision_floors or len(set(precision_floors)) != len(precision_floors):
        raise ValueError("Precision floors must be nonempty and unique.")
    policies = [
        ThresholdPolicy(cutoff_objective, floor, minimum_recall)
        for floor in precision_floors
    ]
    if cutoff_objective == "precision_at_recall":
        policies = [ThresholdPolicy(cutoff_objective, minimum_recall=minimum_recall)]
    candidates = build_candidates(
        tuned=tuned,
        architectures=tuple(architectures or ("hist_gradient_boosting",)),
        feature_sets=tuple(feature_sets or ("ten",)),
        imbalance_strategies=tuple(
            imbalance_strategies or ("balanced_weights", "period_class_weights")
        ),
        seeds=tuple(seeds) if seeds is not None else None,
    )
    identity = input_identity({"development": path})
    data = pd.read_csv(path, float_precision="round_trip")
    return run_grouped_study(
        data, candidates, output, models, plan=plan, policies=policies, inputs=identity
    )

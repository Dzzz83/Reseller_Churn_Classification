from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from reseller_churn.config.feature_sets import (
    FULL_FEATURES,
    PRUNED_FEATURES,
)
from reseller_churn.config.model_settings import (
    MODEL_SEEDS,
    TUNED_RANDOM_FOREST,
    RandomForestSettings,
)
from reseller_churn.config.project_paths import RESULTS_DIR
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.modeling.seeded_predictions import SeededProbabilityPredictor


OUTPUT_DIR = RESULTS_DIR / "model_selection"

ALTERNATIVE_RANDOM_FOREST_SETTINGS = RandomForestSettings(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
)

# These are the features that appear in both:
# 1. our verified 17-feature dataset; and
# 2. the original method's selected "full" feature set.
#
# Current-state store attributes and categorical store fields are deliberately
# excluded because historical availability has not been established.
SHARED_HISTORICAL_FEATURES = [
    "n_orders_6m",
    "n_orders_12m",
    "mean_gap",
    "std_gap",
    "overdue_ratio",
    "share_bikes",
    "share_clothing",
    "share_accessories",
    "store_age",
]


def evaluate_temporal(
    data: pd.DataFrame,
    features: list[str],
    settings: RandomForestSettings,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    temporal_data = TemporalDataset(data)

    fold_scores = []
    pooled_target = []
    pooled_probability = []
    train_sizes = []
    validation_sizes = []
    validation_rates = []

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(fold)

        train = prepared.training_data
        validation = prepared.validation_data

        probabilities_by_seed = SeededProbabilityPredictor.random_forest_by_seed(
            training_features=train[features],
            training_target=train["churn"].astype(int),
            validation_features=validation[features],
            settings=settings,
            seeds=seeds,
        )

        target = validation["churn"].astype(int).to_numpy()

        seed_scores = [
            average_precision_score(
                target,
                probability,
            )
            for probability in probabilities_by_seed
        ]

        fold_scores.append(
            float(np.mean(seed_scores))
        )

        probability = np.mean(
            probabilities_by_seed,
            axis=0,
        )

        pooled_target.extend(target.tolist())
        pooled_probability.extend(probability.tolist())
        train_sizes.append(len(train))
        validation_sizes.append(len(validation))
        validation_rates.append(float(np.mean(target)))

    return {
        "validation_design": "temporal",
        "mean_fold_pr_auc": float(np.mean(fold_scores)),
        "worst_fold_pr_auc": float(np.min(fold_scores)),
        "pooled_oof_pr_auc": float(
            average_precision_score(
                pooled_target,
                pooled_probability,
            )
        ),
        "fold_pr_auc_std": float(
            np.std(fold_scores)
        ),
        "mean_train_rows": float(np.mean(train_sizes)),
        "mean_validation_rows": float(
            np.mean(validation_sizes)
        ),
        "mean_validation_churn_rate": float(
            np.mean(validation_rates)
        ),
    }


def evaluate_group_kfold(
    data: pd.DataFrame,
    features: list[str],
    settings: RandomForestSettings,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    working = data.reset_index(drop=True)

    splitter = GroupKFold(n_splits=5)

    oof_probability = np.zeros(len(working))
    fold_scores = []
    train_sizes = []
    validation_sizes = []
    validation_rates = []

    for train_index, validation_index in splitter.split(
        working,
        working["churn"],
        groups=working["StoreID"],
    ):
        train = working.iloc[train_index]
        validation = working.iloc[validation_index]

        probabilities_by_seed = SeededProbabilityPredictor.random_forest_by_seed(
            training_features=train[features],
            training_target=train["churn"].astype(int),
            validation_features=validation[features],
            settings=settings,
            seeds=seeds,
        )

        probability = np.mean(
            probabilities_by_seed,
            axis=0,
        )

        target = validation["churn"].astype(int).to_numpy()

        oof_probability[validation_index] = probability

        fold_scores.append(
            average_precision_score(
                target,
                probability,
            )
        )

        train_sizes.append(len(train))
        validation_sizes.append(len(validation))
        validation_rates.append(float(np.mean(target)))

    return {
        "validation_design": "GroupKFold(StoreID)",
        "mean_fold_pr_auc": float(np.mean(fold_scores)),
        "worst_fold_pr_auc": float(np.min(fold_scores)),
        "pooled_oof_pr_auc": float(
            average_precision_score(
                working["churn"].astype(int),
                oof_probability,
            )
        ),
        "fold_pr_auc_std": float(
            np.std(fold_scores)
        ),
        "mean_train_rows": float(np.mean(train_sizes)),
        "mean_validation_rows": float(
            np.mean(validation_sizes)
        ),
        "mean_validation_churn_rate": float(
            np.mean(validation_rates)
        ),
    }


def run_experiment(
    data: pd.DataFrame,
    name: str,
    validation_design: str,
    features: list[str],
    settings: RandomForestSettings,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    if validation_design == "temporal":
        metrics = evaluate_temporal(
            data=data,
            features=features,
            settings=settings,
            seeds=seeds,
        )
    elif validation_design == "group":
        metrics = evaluate_group_kfold(
            data=data,
            features=features,
            settings=settings,
            seeds=seeds,
        )
    else:
        raise ValueError(
            f"Unknown validation design: {validation_design}"
        )

    return {
        "experiment": name,
        "feature_count": len(features),
        "seed_count": len(seeds),
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "min_samples_leaf": settings.min_samples_leaf,
        **metrics,
    }


def main() -> None:
    print(
        "=== Model Design Factor Comparison ==="
    )
    print(
        "Development data only. Final-test labels are NOT used."
    )
    print()

    data = DatasetLoader.load_labeled_snapshots()

    experiments = [
        (
            "A_our_baseline_5seeds",
            "temporal",
            PRUNED_FEATURES,
            TUNED_RANDOM_FOREST,
            MODEL_SEEDS,
        ),
        (
            "B_our_rf_seed42_only",
            "temporal",
            PRUNED_FEATURES,
            TUNED_RANDOM_FOREST,
            (42,),
        ),
        (
            "C_correlation_pruned_rf_params_only",
            "temporal",
            PRUNED_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
        (
            "D_correlation_pruned_rf_our_full17",
            "temporal",
            FULL_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
        (
            "E_correlation_pruned_rf_safe_overlap9",
            "temporal",
            SHARED_HISTORICAL_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
        (
            "F_groupkfold_our_rf_pruned7",
            "group",
            PRUNED_FEATURES,
            TUNED_RANDOM_FOREST,
            (42,),
        ),
        (
            "G_groupkfold_correlation_pruned_rf_pruned7",
            "group",
            PRUNED_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
        (
            "H_groupkfold_correlation_pruned_rf_full17",
            "group",
            FULL_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
        (
            "I_groupkfold_correlation_pruned_rf_overlap9",
            "group",
            SHARED_HISTORICAL_FEATURES,
            ALTERNATIVE_RANDOM_FOREST_SETTINGS,
            (42,),
        ),
    ]

    rows = []

    for (
        name,
        validation_design,
        features,
        settings,
        seeds,
    ) in experiments:
        print(f"Running {name}...")

        rows.append(
            run_experiment(
                data=data,
                name=name,
                validation_design=validation_design,
                features=features,
                settings=settings,
                seeds=seeds,
            )
        )

    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "02_model_design_factor_comparison.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    display_columns = [
        "experiment",
        "validation_design",
        "feature_count",
        "seed_count",
        "mean_fold_pr_auc",
        "worst_fold_pr_auc",
        "pooled_oof_pr_auc",
        "fold_pr_auc_std",
        "mean_train_rows",
        "mean_validation_churn_rate",
    ]

    print()
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print("Reference implementation:")
    print(
        "Correlation-pruned full RF: 5-fold GroupKFold by StoreID, "
        "1,620 training observations, 20 selected features, "
        "pooled OOF PR-AUC = 0.500."
    )
    print(
        "That number is NOT directly comparable to the rows above "
        "because the snapshot schedule, feature construction, and "
        "training population are different."
    )
    print()
    print(
        "Important: GroupKFold rows are diagnostic only. "
        "They are not candidates for replacing temporal validation."
    )
    print(
        "Final-test labels were NOT used."
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.model_settings import (
    GradientBoostingSettings,
    MODEL_SEEDS,
    RandomForestSettings,
)
from reseller_churn.config.project_paths import RESULTS_DIR
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.resampling import (
    balance_classes_by_random_oversampling,
)


OUTPUT_DIR = RESULTS_DIR / "hyperparameter_tuning"

RF_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", 0.7],
}

GB_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 200],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [10, 20, 30],
    "l2_regularization": [0.0, 1.0],
}


def tune_random_forest(
    temporal_data: TemporalDataset,
) -> pd.DataFrame:
    rows = []

    for parameters in ParameterGrid(RF_GRID):
        fold_scores = []

        for fold in DEVELOPMENT_FOLDS:
            prepared = temporal_data.prepare_fold(fold)

            x_train = prepared.training_data[
                PRUNED_FEATURES
            ]
            y_train = prepared.training_data[
                "churn"
            ].astype(int)
            x_val = prepared.validation_data[
                PRUNED_FEATURES
            ]
            y_val = prepared.validation_data[
                "churn"
            ].astype(int)

            settings = RandomForestSettings(
                **parameters
            )

            seed_scores = []

            for seed in MODEL_SEEDS:
                model = ModelFactory.create_random_forest(
                    settings,
                    seed,
                )

                model.fit(
                    x_train,
                    y_train,
                )

                seed_scores.append(
                    average_precision_score(
                        y_val,
                        model.predict_proba(x_val)[:, 1],
                    )
                )

            fold_scores.append(
                float(np.mean(seed_scores))
            )

        rows.append(
            {
                "model": "Random Forest",
                "params": json.dumps(parameters),
                "fold_1_pr_auc": fold_scores[0],
                "fold_2_pr_auc": fold_scores[1],
                "mean_pr_auc": float(np.mean(fold_scores)),
                "worst_fold_pr_auc": float(np.min(fold_scores)),
            }
        )

    return pd.DataFrame(rows)


def tune_gradient_boosting(
    temporal_data: TemporalDataset,
) -> pd.DataFrame:
    rows = []

    for parameters in ParameterGrid(GB_GRID):
        fold_scores = []

        for fold in DEVELOPMENT_FOLDS:
            prepared = temporal_data.prepare_fold(fold)

            x_train = prepared.training_data[
                PRUNED_FEATURES
            ]
            y_train = prepared.training_data[
                "churn"
            ].astype(int)
            x_val = prepared.validation_data[
                PRUNED_FEATURES
            ]
            y_val = prepared.validation_data[
                "churn"
            ].astype(int)

            settings = GradientBoostingSettings(
                **parameters
            )

            seed_scores = []

            for seed in MODEL_SEEDS:
                x_balanced, y_balanced = (
                    balance_classes_by_random_oversampling(
                        features=x_train,
                        target=y_train,
                        random_seed=seed,
                    )
                )

                model = ModelFactory.create_gradient_boosting(
                    settings,
                    seed,
                )

                model.fit(
                    x_balanced,
                    y_balanced,
                )

                seed_scores.append(
                    average_precision_score(
                        y_val,
                        model.predict_proba(x_val)[:, 1],
                    )
                )

            fold_scores.append(
                float(np.mean(seed_scores))
            )

        rows.append(
            {
                "model": "Gradient Boosting",
                "params": json.dumps(parameters),
                "fold_1_pr_auc": fold_scores[0],
                "fold_2_pr_auc": fold_scores[1],
                "mean_pr_auc": float(np.mean(fold_scores)),
                "worst_fold_pr_auc": float(np.min(fold_scores)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=== 09. Hyperparameter Tuning ===")

    temporal_data = TemporalDataset(
        DatasetLoader.load_labeled_snapshots()
    )

    print("Tuning Random Forest...")
    rf_results = tune_random_forest(
        temporal_data
    ).sort_values(
        [
            "mean_pr_auc",
            "worst_fold_pr_auc",
        ],
        ascending=False,
    )

    print("Tuning Gradient Boosting...")
    gb_results = tune_gradient_boosting(
        temporal_data
    ).sort_values(
        [
            "mean_pr_auc",
            "worst_fold_pr_auc",
        ],
        ascending=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rf_results.to_csv(
        OUTPUT_DIR / "random_forest_tuning.csv",
        index=False,
    )

    gb_results.to_csv(
        OUTPUT_DIR / "gradient_boosting_tuning.csv",
        index=False,
    )

    print()
    print("Top Random Forest")
    print(
        rf_results.head(5).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print("Top Gradient Boosting")
    print(
        gb_results.head(5).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

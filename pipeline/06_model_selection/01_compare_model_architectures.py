from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from reseller_churn.config.feature_sets import REDUCED_FEATURES
from reseller_churn.config.model_settings import (
    ARCHITECTURE_COMPARISON_SEEDS,
    BASELINE_GRADIENT_BOOSTING,
    BASELINE_RANDOM_FOREST,
)
from reseller_churn.config.project_paths import RESULTS_DIR
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.evaluation.metrics import calculate_ranking_metrics


OUTPUT_DIR = RESULTS_DIR / "model_selection"


def score_model(model, x_train, y_train, x_val, y_val) -> dict[str, float]:
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]

    return calculate_ranking_metrics(y_val, probabilities)


def main() -> None:
    print("=== 06. Model Architecture Selection ===")

    data = DatasetLoader.load_labeled_snapshots()
    temporal_data = TemporalDataset(data)

    rows = []

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(fold)

        x_train = prepared.training_data[
            REDUCED_FEATURES
        ]
        y_train = prepared.training_data[
            "churn"
        ].astype(int)

        x_val = prepared.validation_data[
            REDUCED_FEATURES
        ]
        y_val = prepared.validation_data[
            "churn"
        ].astype(int)

        logistic = ModelFactory.create_logistic_regression()

        rows.append(
            {
                "fold": fold.name,
                "model": "Logistic Regression",
                "seed": np.nan,
                **score_model(
                    logistic,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                ),
            }
        )

        for seed in ARCHITECTURE_COMPARISON_SEEDS:
            rf = ModelFactory.create_random_forest(
                BASELINE_RANDOM_FOREST,
                seed,
            )

            rows.append(
                {
                    "fold": fold.name,
                    "model": "Random Forest",
                    "seed": seed,
                    **score_model(
                        rf,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                    ),
                }
            )

            gb = ModelFactory.create_gradient_boosting(
                BASELINE_GRADIENT_BOOSTING,
                seed,
            )

            rows.append(
                {
                    "fold": fold.name,
                    "model": "Gradient Boosting",
                    "seed": seed,
                    **score_model(
                        gb,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                    ),
                }
            )

    results = pd.DataFrame(rows)

    per_fold = (
        results.groupby(
            ["fold", "model"]
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )

    per_fold["std_pr_auc"] = (
        per_fold["std_pr_auc"].fillna(0)
    )

    summary = (
        per_fold.groupby("model")
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold_pr_auc=("mean_pr_auc", "min"),
            mean_roc_auc=("mean_roc_auc", "mean"),
        )
        .reset_index()
        .sort_values(
            "mean_pr_auc",
            ascending=False,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_DIR / "01_architecture_runs.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "01_architecture_summary.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )
    print()
    print("Primary decision metric: PR-AUC")
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

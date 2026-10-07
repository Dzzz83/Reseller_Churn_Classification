from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import average_precision_score

from reseller_churn.step_00_config.feature_sets import FEATURE_SETS
from reseller_churn.step_00_config.model_settings import (
    ARCHITECTURE_COMPARISON_SEEDS,
    BASELINE_GRADIENT_BOOSTING,
    BASELINE_RANDOM_FOREST,
)
from reseller_churn.step_00_config.project_paths import RESULTS_DIR
from reseller_churn.step_00_config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_01_data.temporal_dataset import TemporalDataset
from reseller_churn.step_03_modeling.model_factory import ModelFactory


OUTPUT_DIR = RESULTS_DIR / "feature_selection"


def main() -> None:
    print("=== 07. Feature-Set Selection ===")

    data = DatasetLoader.load_labeled_snapshots()
    temporal_data = TemporalDataset(data)

    rows = []

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(fold)

        for feature_set_name, feature_names in FEATURE_SETS.items():
            x_train = prepared.training_data[
                feature_names
            ]
            y_train = prepared.training_data[
                "churn"
            ].astype(int)

            x_val = prepared.validation_data[
                feature_names
            ]
            y_val = prepared.validation_data[
                "churn"
            ].astype(int)

            rf_scores = []

            for seed in ARCHITECTURE_COMPARISON_SEEDS:
                rf = ModelFactory.create_random_forest(
                    BASELINE_RANDOM_FOREST,
                    seed,
                )
                rf.fit(x_train, y_train)

                rf_scores.append(
                    average_precision_score(
                        y_val,
                        rf.predict_proba(x_val)[:, 1],
                    )
                )

            gb = ModelFactory.create_gradient_boosting(
                BASELINE_GRADIENT_BOOSTING,
                42,
            )
            gb.fit(x_train, y_train)

            rows.extend(
                [
                    {
                        "fold": fold.name,
                        "model": "Random Forest",
                        "feature_set": feature_set_name,
                        "pr_auc": sum(rf_scores) / len(rf_scores),
                    },
                    {
                        "fold": fold.name,
                        "model": "Gradient Boosting",
                        "feature_set": feature_set_name,
                        "pr_auc": average_precision_score(
                            y_val,
                            gb.predict_proba(x_val)[:, 1],
                        ),
                    },
                ]
            )

    per_fold = pd.DataFrame(rows)

    summary = (
        per_fold.groupby(
            ["model", "feature_set"]
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            worst_fold_pr_auc=("pr_auc", "min"),
        )
        .reset_index()
        .sort_values(
            ["model", "mean_pr_auc"],
            ascending=[True, False],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_fold.to_csv(
        OUTPUT_DIR / "01_feature_set_folds.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "01_feature_set_summary.csv",
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
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

from reseller_churn.config.feature_sets import PRUNED_FEATURES
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
from reseller_churn.modeling.resampling import (
    balance_classes_by_random_oversampling,
)
from reseller_churn.evaluation.metrics import calculate_threshold_metrics


OUTPUT_DIR = RESULTS_DIR / "imbalance_strategy"
DIAGNOSTIC_THRESHOLD = 0.5


def score(target, probabilities) -> dict[str, float]:
    threshold_metrics = calculate_threshold_metrics(
        target,
        probabilities,
        DIAGNOSTIC_THRESHOLD,
    )
    return {
        "pr_auc": average_precision_score(target, probabilities),
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
    }


def main() -> None:
    print("=== 08. Imbalance Strategy Comparison ===")

    data = DatasetLoader.load_labeled_snapshots()
    temporal_data = TemporalDataset(data)

    rows = []

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

        balanced_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        for seed in ARCHITECTURE_COMPARISON_SEEDS:
            for strategy in (
                "none",
                "balanced_weights",
                "random_oversampling",
            ):
                model = ModelFactory.create_random_forest(
                    BASELINE_RANDOM_FOREST,
                    seed,
                )

                current_x = x_train
                current_y = y_train
                fit_kwargs = {}

                if strategy == "balanced_weights":
                    fit_kwargs[
                        "model__sample_weight"
                    ] = balanced_weights

                if strategy == "random_oversampling":
                    current_x, current_y = (
                        balance_classes_by_random_oversampling(
                            features=x_train,
                            target=y_train,
                            random_seed=seed,
                        )
                    )

                model.fit(
                    current_x,
                    current_y,
                    **fit_kwargs,
                )

                rows.append(
                    {
                        "fold": fold.name,
                        "model": "Random Forest",
                        "strategy": strategy,
                        "seed": seed,
                        **score(
                            y_val,
                            model.predict_proba(x_val)[:, 1],
                        ),
                    }
                )

        for strategy in (
            "none",
            "balanced_weights",
            "random_oversampling",
        ):
            seed = 42
            model = ModelFactory.create_gradient_boosting(
                BASELINE_GRADIENT_BOOSTING,
                seed,
            )

            current_x = x_train
            current_y = y_train
            fit_kwargs = {}

            if strategy == "balanced_weights":
                fit_kwargs["sample_weight"] = (
                    balanced_weights
                )

            if strategy == "random_oversampling":
                current_x, current_y = (
                    balance_classes_by_random_oversampling(
                        features=x_train,
                        target=y_train,
                        random_seed=seed,
                    )
                )

            model.fit(
                current_x,
                current_y,
                **fit_kwargs,
            )

            rows.append(
                {
                    "fold": fold.name,
                    "model": "Gradient Boosting",
                    "strategy": strategy,
                    "seed": np.nan,
                    **score(
                        y_val,
                        model.predict_proba(x_val)[:, 1],
                    ),
                }
            )

    results = pd.DataFrame(rows)

    per_fold = (
        results.groupby(
            ["model", "strategy", "fold"]
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_fold.groupby(
            ["model", "strategy"]
        )
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold_pr_auc=("mean_pr_auc", "min"),
            mean_precision=("mean_precision", "mean"),
            mean_recall=("mean_recall", "mean"),
            mean_f1=("mean_f1", "mean"),
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

    results.to_csv(
        OUTPUT_DIR / "01_strategy_runs.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "01_strategy_summary.csv",
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
    print(
        "Random oversampling is applied to training folds only."
    )
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

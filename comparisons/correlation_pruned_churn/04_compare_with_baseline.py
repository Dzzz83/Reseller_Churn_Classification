"""Match the two methods on identical snapshots, resellers and labels.

The primary project's already-tuned Random Forest is not retuned here.
Both models are evaluated on the same corrected development data and the
same two explicit validation designs. Final-test labels stay untouched.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons.correlation_pruned_churn.corrected_features import HistoricalSnapshotBuilder
from comparisons.correlation_pruned_churn.validation import (
    DEVELOPMENT_SNAPSHOTS,
    ValidationPlans,
)
from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.model_settings import TUNED_RANDOM_FOREST
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.features.feature_pipeline import FeaturePipeline
from reseller_churn.modeling.model_factory import ModelFactory


OUTPUT = ROOT / "results" / "correlation_pruned_churn"


def main() -> None:
    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    labels = HistoricalSnapshotBuilder(orders, stores).build(
        DEVELOPMENT_SNAPSHOTS,
        feature_window="full",
    )[["StoreID", "snapshot", "churn"]]

    features = FeaturePipeline().build(
        orders,
        stores,
        list(DEVELOPMENT_SNAPSHOTS),
    )
    features["snapshot"] = (
        pd.to_datetime(features["snapshot"])
        .dt.strftime("%Y-%m-%d")
    )
    data = features.merge(
        labels,
        on=["StoreID", "snapshot"],
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(labels):
        raise AssertionError(
            "Different eligible reseller populations"
        )
    if data.duplicated(["StoreID", "snapshot"]).any():
        raise AssertionError("Duplicated observation key")

    # Validate the entire key set, not merely the same number of rows.
    expected_keys = set(
        zip(labels["StoreID"], labels["snapshot"])
    )
    actual_keys = set(
        zip(data["StoreID"], data["snapshot"])
    )
    if actual_keys != expected_keys:
        raise AssertionError("Unmatched reseller/snapshot rows")

    results = []
    for strategy, folds in (
        ("temporal", ValidationPlans.temporal(data)),
        ("group_kfold", ValidationPlans.group_kfold(data)),
    ):
        scores = []
        truth = []
        predictions = []
        for fold in folds:
            train = data.iloc[fold.training_indices]
            validation = data.iloc[fold.validation_indices]

            model = ModelFactory.create_random_forest(
                TUNED_RANDOM_FOREST,
                random_seed=42,
            )
            model.fit(
                train[PRUNED_FEATURES],
                train["churn"].astype(int),
            )
            probability = model.predict_proba(
                validation[PRUNED_FEATURES]
            )[:, 1]

            y_val = validation["churn"].astype(int)
            scores.append(
                average_precision_score(y_val, probability)
            )
            truth.extend(y_val.tolist())
            predictions.extend(probability.tolist())

        results.append(
            {
                "variant": "our_12m_pruned7",
                "feature_scope": "historical_only",
                "strategy": strategy,
                "model": "RandomForest_tuned_seed42",
                "folds": len(scores),
                "mean_pr_auc": float(np.mean(scores)),
                "worst_fold_pr_auc": float(np.min(scores)),
                "pooled_oof_pr_auc": average_precision_score(
                    truth, predictions
                ),
            }
        )

    comparison = pd.DataFrame(results)

    comparison_path = OUTPUT / "03_model_comparison.csv"
    if comparison_path.exists():
        correlation_pruned_scores = pd.read_csv(comparison_path)
        correlation_pruned_scores = correlation_pruned_scores[
            (correlation_pruned_scores["feature_scope"] == "historical_only")
            & (correlation_pruned_scores["model"] == "RandomForest")
        ]
        shared_cols = list(comparison.columns)
        comparison = pd.concat(
            [
                comparison,
                correlation_pruned_scores[shared_cols],
            ],
            ignore_index=True,
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "04_matched_model_comparison.csv"
    comparison.to_csv(path, index=False)

    print("=== Matched Development Comparison ===")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print(
        "Same snapshots, eligible resellers, churn labels, "
        "and validation strategies."
    )
    print(
        "Models retain their own feature sets and RF parameters. "
        "Differences cannot be assigned to a single factor."
    )
    print("Final-test labels NOT accessed.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

"""Evaluate both legitimate validation questions with fold-local feature fitting.

Runs only historical development snapshots. Does not tune thresholds or read
final-test churn outcomes.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from comparisons.correlation_pruned_churn.corrected_features import HistoricalSnapshotBuilder
from comparisons.correlation_pruned_churn.corrected_model import fit_correlation_pruned_model
from comparisons.correlation_pruned_churn.feature_provenance import attach_feature_scope
from comparisons.correlation_pruned_churn.validation import (
    DEVELOPMENT_SNAPSHOTS,
    ValidationPlans,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


OUTPUT = ROOT / "results" / "correlation_pruned_churn"


def evaluate(
    frame: pd.DataFrame,
    numeric_candidates: tuple[str, ...],
    categorical_features: tuple[str, ...],
    variant: str,
    profile_scope: str,
    strategy: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "temporal":
        folds = ValidationPlans.temporal(frame)
    elif strategy == "group_kfold":
        folds = ValidationPlans.group_kfold(frame)
    else:
        raise ValueError(f"Unknown validation design: {strategy}")

    summary_rows = []
    prediction_rows = []

    for fold in folds:
        train = frame.iloc[fold.training_indices]
        validation = frame.iloc[fold.validation_indices]

        model, features = fit_correlation_pruned_model(
            train=train,
            numeric_candidates=list(numeric_candidates),
            model_name=model_name,
            categorical_features=categorical_features,
        )

        probability = model.predict_proba(
            validation[features]
        )[:, 1]
        truth = validation["churn"].astype(int).to_numpy()
        predicted = (probability >= 0.5).astype(int)

        if len(np.unique(truth)) != 2:
            raise AssertionError(
                f"{fold.name}: validation labels have only one class"
            )

        summary_rows.append(
            {
                "variant": variant,
                "feature_scope": profile_scope,
                "strategy": strategy,
                "model": model_name,
                "fold": fold.name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "validation_churners": int(truth.sum()),
                "validation_churn_rate": float(truth.mean()),
                "selected_feature_count": len(features),
                "selected_features": ",".join(features),
                "pr_auc": average_precision_score(truth, probability),
                "roc_auc": roc_auc_score(truth, probability),
                "precision_at_0_5": precision_score(
                    truth, predicted, zero_division=0
                ),
                "recall_at_0_5": recall_score(
                    truth, predicted, zero_division=0
                ),
                "f1_at_0_5": f1_score(
                    truth, predicted, zero_division=0
                ),
            }
        )

        for store, snapshot, label, score in zip(
            validation["StoreID"],
            validation["snapshot"],
            truth,
            probability,
        ):
            prediction_rows.append(
                {
                    "variant": variant,
                    "feature_scope": profile_scope,
                    "strategy": strategy,
                    "model": model_name,
                    "fold": fold.name,
                    "StoreID": int(store),
                    "snapshot": snapshot,
                    "churn": int(label),
                    "probability": float(score),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def summarize(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    by = ["variant", "feature_scope", "strategy", "model"]
    group_rows = []

    for group_values, sub in folds.groupby(by, sort=False):
        mask = np.ones(len(predictions), dtype=bool)
        for field, value in zip(by, group_values):
            mask &= predictions[field].eq(value).to_numpy()
        oof = predictions.loc[mask]

        group_rows.append(
            {
                **dict(zip(by, group_values)),
                "folds": len(sub),
                "mean_pr_auc": sub["pr_auc"].mean(),
                "worst_fold_pr_auc": sub["pr_auc"].min(),
                "fold_pr_auc_std": sub["pr_auc"].std(ddof=0),
                "pooled_oof_pr_auc": average_precision_score(
                    oof["churn"],
                    oof["probability"],
                ),
                "mean_roc_auc": sub["roc_auc"].mean(),
                "mean_precision_at_0_5": sub["precision_at_0_5"].mean(),
                "mean_recall_at_0_5": sub["recall_at_0_5"].mean(),
                "mean_f1_at_0_5": sub["f1_at_0_5"].mean(),
                "mean_train_rows": sub["train_rows"].mean(),
                "mean_validation_churn_rate": (
                    sub["validation_churn_rate"].mean()
                ),
            }
        )

    return pd.DataFrame(group_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-unverified-store-profile",
        action="store_true",
        help=(
            "Run an ADDITIONAL diagnostic with current store fields "
            "whose historical availability has not been verified."
        ),
    )
    args = parser.parse_args()

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    builder = HistoricalSnapshotBuilder(orders, stores)
    variants = {
        name: builder.build(
            DEVELOPMENT_SNAPSHOTS,
            feature_window=name,
        )
        for name in ("full", "obs6")
    }

    print("=== Corrected Friend Method — Development Only ===", flush=True)
    print(
        "GroupKFold = unseen reseller identity. "
        "Temporal = future-period prediction.",
        flush=True,
    )
    print("All feature selection fits inside training folds.", flush=True)
    print("Final-test labels NOT accessed.", flush=True)

    all_fold_metrics = []
    all_predictions = []

    for variant, raw in variants.items():
        include_profiles = (
            (False, True)
            if args.include_unverified_store_profile
            else (False,)
        )
        for include_profile in include_profiles:
            frame, scope = attach_feature_scope(
                raw, stores, include_profile
            )

            for strategy in ("temporal", "group_kfold"):
                for model_name in ("RandomForest", "LogReg"):
                    print(
                        f"{variant} / {scope.name} / {strategy} / "
                        f"{model_name}",
                        flush=True,
                    )
                    folds, predictions = evaluate(
                        frame=frame,
                        numeric_candidates=scope.numeric_candidates,
                        categorical_features=scope.categorical_features,
                        variant=variant,
                        profile_scope=scope.name,
                        strategy=strategy,
                        model_name=model_name,
                    )
                    all_fold_metrics.append(folds)
                    all_predictions.append(predictions)

    fold_metrics = pd.concat(
        all_fold_metrics, ignore_index=True
    )
    predictions = pd.concat(
        all_predictions, ignore_index=True
    )
    overview = summarize(fold_metrics, predictions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(
        OUTPUT / "03_fold_metrics.csv", index=False
    )
    overview.to_csv(
        OUTPUT / "03_model_comparison.csv", index=False
    )
    predictions.to_csv(
        OUTPUT / "03_oof_predictions.csv", index=False
    )

    display = [
        "variant",
        "feature_scope",
        "strategy",
        "model",
        "mean_pr_auc",
        "worst_fold_pr_auc",
        "pooled_oof_pr_auc",
        "mean_train_rows",
    ]
    print()
    print("=== Development Comparison ===")
    print(
        overview[display].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("Threshold 0.5 diagnostics are NOT tuned or locked.")
    print("GroupKFold and temporal scores measure different questions.")
    print("No final-test labels were used.")
    print(f"Saved under: {OUTPUT}")


if __name__ == "__main__":
    main()

"""Reproduce the original 6-6 GroupKFold experiment with audited corrections.

Keep the original five training snapshots, November 2013 retrospective test,
model settings and OOF F1 threshold-selection rule. Corrections:
- half-open six-month churn horizon;
- supervised correlation pruning separately within each CV training fold;
- an explicit historical-only versus unverified-store-profile comparison.

No October 2013 final-test labels or artifacts are accessed.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from comparisons.friend_method.corrected_features import FriendSnapshotBuilder
from comparisons.friend_method.corrected_model import fit_friend_model
from comparisons.friend_method.feature_provenance import attach_feature_scope
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


TRAIN_SNAPSHOTS = (
    "2012-05-01", "2012-08-01", "2012-11-01",
    "2013-02-01", "2013-05-01",
)
RETROSPECTIVE_TEST = "2013-11-01"
OUTPUT = ROOT / "results" / "friend_method_audit"

# Exact metrics transcribed from phase2.ipynb, code-cell 15.
# These are the *published, uncorrected* reference scores, not experiment inputs.
PUBLISHED = {
    ("full", "LogReg"): (0.378, 0.609, 0.304, 0.346, 0.204),
    ("full", "RandomForest"): (0.500, 0.705, 0.518, 0.846, 0.231),
    ("obs6", "LogReg"): (0.404, 0.605, 0.283, 0.885, 0.183),
    ("obs6", "RandomForest"): (0.469, 0.694, 0.380, 0.962, 0.203),
}


def threshold_maximizing_oof_f1(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Faithfully reproduce the original notebook's threshold selection."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def validate_protocol(training: pd.DataFrame, test: pd.DataFrame) -> None:
    assert set(training["snapshot"]) == set(TRAIN_SNAPSHOTS)
    assert set(test["snapshot"]) == {RETROSPECTIVE_TEST}
    assert not training.duplicated(["StoreID", "snapshot"]).any()
    assert not test.duplicated(["StoreID", "snapshot"]).any()
    for date in TRAIN_SNAPSHOTS:
        assert pd.Timestamp(date) + pd.DateOffset(months=6) <= pd.Timestamp(
            RETROSPECTIVE_TEST
        )
    assert len(test) == 475, "November test cohort differs from original"
    assert int(test["churn"].sum()) == 35, "Half-open November churn labels differ"


def evaluate_groupkfold(
    training: pd.DataFrame,
    test: pd.DataFrame,
    numeric_candidates: tuple[str, ...],
    categorical_features: tuple[str, ...],
    variant: str,
    scope: str,
    model_name: str,
) -> tuple[dict, list[dict], list[dict]]:
    y = training["churn"].astype(int).to_numpy()
    oof = np.full(len(training), np.nan)
    folds = []
    splitter = GroupKFold(n_splits=5)

    for fold_id, (train_indices, validation_indices) in enumerate(
        splitter.split(training, y, groups=training["StoreID"]), start=1
    ):
        fold_train = training.iloc[train_indices]
        fold_valid = training.iloc[validation_indices]
        assert set(fold_train["StoreID"]).isdisjoint(set(fold_valid["StoreID"]))

        # Crucial difference from original: churn-based pruning only sees fold_train.
        model, columns = fit_friend_model(
            fold_train, list(numeric_candidates), model_name, categorical_features
        )
        predictions = model.predict_proba(fold_valid[columns])[:, 1]
        oof[validation_indices] = predictions
        fold_metrics = classification_metrics(
            fold_valid["churn"].astype(int).to_numpy(), predictions, 0.5
        )
        folds.append({
            "variant": variant, "feature_scope": scope, "model": model_name,
            "fold": fold_id,
            "train_rows": len(fold_train), "validation_rows": len(fold_valid),
            "validation_churners": int(fold_valid["churn"].sum()),
            "selected_features": ",".join(columns),
            "pr_auc": fold_metrics["pr_auc"], "roc_auc": fold_metrics["roc_auc"],
        })

    assert np.isfinite(oof).all(), "Missing GroupKFold predictions"
    # Original rule: select threshold on the complete training OOF predictions.
    # Recall/F1 on those same OOF rows is descriptive and optimistically selected.
    threshold = threshold_maximizing_oof_f1(y, oof)
    oof_scores = classification_metrics(y, oof, threshold)

    # Refit feature selection and preprocessing from scratch on all five train
    # snapshots; no November test labels influence fitting or threshold choice.
    final_model, final_columns = fit_friend_model(
        training, list(numeric_candidates), model_name, categorical_features
    )
    y_test = test["churn"].astype(int).to_numpy()
    test_probabilities = final_model.predict_proba(test[final_columns])[:, 1]
    test_scores = classification_metrics(y_test, test_probabilities, threshold)

    summary = {
        "variant": variant, "feature_scope": scope, "model": model_name,
        "train_rows": len(training), "train_churners": int(y.sum()),
        "test_rows": len(test), "test_churners": int(y_test.sum()),
        "threshold": threshold,
        "cv_mean_fold_pr_auc": float(np.mean([fold["pr_auc"] for fold in folds])),
        "cv_std_fold_pr_auc": float(np.std([fold["pr_auc"] for fold in folds])),
        **{"cv_" + name: value for name, value in oof_scores.items()},
        **{"test_" + name: value for name, value in test_scores.items()},
        "final_selected_features": ",".join(final_columns),
    }

    predictions = []
    for subset, frame, actuals, probs in (
        ("cv_oof", training, y, oof),
        ("november_retrospective", test, y_test, test_probabilities),
    ):
        predictions.extend({
            "variant": variant, "feature_scope": scope, "model": model_name,
            "subset": subset, "StoreID": int(store),
            "snapshot": snapshot, "churn": int(actual),
            "probability": float(score), "threshold": threshold,
        } for store, snapshot, actual, score in zip(
            frame["StoreID"], frame["snapshot"], actuals, probs
        ))
    return summary, folds, predictions


def make_published_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    # The historically unverified profile scope resembles the original notebook
    # most closely. Its comparison is a diagnostic, NOT a safe deployment claim.
    subset = summary.loc[
        summary["feature_scope"] == "unverified_store_profile_diagnostic"
    ].copy()
    for field, position in (
        ("original_cv_pr_auc", 0), ("original_cv_recall", 1),
        ("original_test_pr_auc", 2), ("original_test_recall", 3),
        ("original_threshold", 4),
    ):
        subset[field] = [
            PUBLISHED[(row.variant, row.model)][position]
            for row in subset.itertuples()
        ]
    for field in ("cv_pr_auc", "cv_recall", "test_pr_auc", "test_recall"):
        subset[field + "_delta"] = subset[field] - subset["original_" + field]
    return subset


def main() -> None:
    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()
    assert len(orders) == 3800, "Expected cleaned 3,800-order source"
    builder = FriendSnapshotBuilder(orders, stores)
    summary_rows, fold_rows, prediction_rows = [], [], []

    for variant in ("full", "obs6"):
        train_raw = builder.build(TRAIN_SNAPSHOTS, feature_window=variant)
        test_raw = builder.build((RETROSPECTIVE_TEST,), feature_window=variant)
        validate_protocol(train_raw, test_raw)

        for include_unverified in (False, True):
            train, scope = attach_feature_scope(
                train_raw, stores, include_unverified
            )
            test, test_scope = attach_feature_scope(
                test_raw, stores, include_unverified
            )
            assert scope == test_scope

            for model_name in ("LogReg", "RandomForest"):
                print(
                    f"RUN {variant} / {scope.name} / {model_name}", flush=True
                )
                summary, folds, predictions = evaluate_groupkfold(
                    train, test, scope.numeric_candidates,
                    scope.categorical_features, variant, scope.name, model_name
                )
                summary_rows.append(summary)
                fold_rows.extend(folds)
                prediction_rows.extend(predictions)

    summary = pd.DataFrame(summary_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "05_original_reproduction_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT / "05_original_reproduction_folds.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT / "05_original_reproduction_predictions.csv", index=False
    )
    comparison = make_published_comparison(summary)
    comparison.to_csv(
        OUTPUT / "05_original_vs_corrected.csv", index=False
    )

    display = [
        "variant", "feature_scope", "model", "threshold",
        "cv_pr_auc", "cv_recall", "cv_precision", "cv_f1",
        "test_pr_auc", "test_recall", "test_precision", "test_f1",
    ]
    print("\n=== Corrected five-snapshot GroupKFold reproduction ===")
    print(summary[display].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Published original versus corrected diagnostic ===")
    comparison_columns = [
        "variant", "model", "original_cv_recall", "cv_recall",
        "original_test_recall", "test_recall",
        "original_cv_pr_auc", "cv_pr_auc",
        "original_test_pr_auc", "test_pr_auc",
    ]
    print(comparison[comparison_columns].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    ))
    print("\nWARNING: CV recall/F1 use OOF-F1-selected thresholds on those same OOF rows.")
    print("WARNING: Profile diagnostics are not proven historically leakage-free.")
    print("WARNING: November is a previously inspected retrospective test.")
    print("The protected October final-test outcomes were NOT accessed.")


if __name__ == "__main__":
    main()

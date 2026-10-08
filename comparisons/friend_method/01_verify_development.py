"""Run the corrected friend's two feature variants on chronological folds.

IMPORTANT: This script deliberately does NOT construct any observations
whose churn labels extend beyond 2013-10-01. It never reads either project's
final-test labels.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from comparisons.friend_method.corrected_features import (
    FriendSnapshotBuilder,
    NUMERIC_FEATURES,
)
from comparisons.friend_method.corrected_model import fit_friend_model
from reseller_churn.step_00_config.project_paths import RESULTS_DIR
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


OUTPUT_DIR = RESULTS_DIR / "friend_method_audit"
LAST_ALLOWED_LABEL_END = pd.Timestamp("2013-10-01")

DEVELOPMENT_SNAPSHOTS = (
    "2012-05-01",
    "2012-08-01",
    "2012-11-01",
    "2013-02-01",
)

TEMPORAL_FOLDS = (
    (
        "fold_1",
        ("2012-05-01",),
        "2012-11-01",
    ),
    (
        "fold_2",
        ("2012-05-01", "2012-08-01"),
        "2013-02-01",
    ),
)


def verify_date_boundaries() -> None:
    for snapshot in DEVELOPMENT_SNAPSHOTS:
        end = (
            pd.Timestamp(snapshot)
            + pd.DateOffset(months=6)
        )
        if end > LAST_ALLOWED_LABEL_END:
            raise AssertionError(
                "Development label extends beyond "
                "protected 2013-10-01 test snapshot"
            )

    for _, train_snapshots, validation_snapshot in TEMPORAL_FOLDS:
        validation_date = pd.Timestamp(validation_snapshot)
        for training_snapshot in train_snapshots:
            if (
                pd.Timestamp(training_snapshot)
                + pd.DateOffset(months=6)
                > validation_date
            ):
                raise AssertionError(
                    "Fold uses a training label that is unknown "
                    "at its validation snapshot"
                )


def run_fold(
    frame: pd.DataFrame,
    variant: str,
    model_name: str,
    fold_name: str,
    train_snapshots: tuple[str, ...],
    validation_snapshot: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train = frame[
        frame["snapshot"].isin(train_snapshots)
    ].copy()

    validation = frame[
        frame["snapshot"] == validation_snapshot
    ].copy()

    numeric_candidates = [
        feature for feature in NUMERIC_FEATURES
        if feature in frame.columns
    ]
    model, selected_features = fit_friend_model(
        train=train,
        numeric_candidates=numeric_candidates,
        model_name=model_name,
    )

    truth = validation["churn"].astype(int).to_numpy()
    probabilities = model.predict_proba(
        validation[selected_features]
    )[:, 1]
    predicted_at_half = (probabilities >= 0.5).astype(int)

    metrics = {
        "variant": variant,
        "model": model_name,
        "fold": fold_name,
        "training_snapshots": ",".join(train_snapshots),
        "validation_snapshot": validation_snapshot,
        "train_rows": len(train),
        "train_churners": int(train["churn"].sum()),
        "validation_rows": len(validation),
        "validation_churners": int(truth.sum()),
        "validation_churn_rate": float(truth.mean()),
        "feature_count": len(selected_features),
        "selected_features": ",".join(selected_features),
        "pr_auc": average_precision_score(truth, probabilities),
        "roc_auc": roc_auc_score(truth, probabilities),
        "precision_at_0_5": precision_score(
            truth, predicted_at_half, zero_division=0
        ),
        "recall_at_0_5": recall_score(
            truth, predicted_at_half, zero_division=0
        ),
        "f1_at_0_5": f1_score(
            truth, predicted_at_half, zero_division=0
        ),
    }

    predictions = pd.DataFrame(
        {
            "variant": variant,
            "model": model_name,
            "fold": fold_name,
            "StoreID": validation["StoreID"].to_numpy(),
            "snapshot": validation["snapshot"].to_numpy(),
            "actual_churn": truth,
            "churn_probability": probabilities,
        }
    )

    return metrics, predictions


def main() -> None:
    verify_date_boundaries()

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    if len(orders) != 3800 or len(stores) != 701:
        raise AssertionError(
            "Expected verified 3,800 orders and 701 stores"
        )

    if (
        (orders["SubTotal"] <= 0).any()
        or orders["qty"].isna().any()
        or (orders["qty"] <= 0).any()
    ):
        raise AssertionError("Invalid orders were not removed")

    builder = FriendSnapshotBuilder(orders=orders, stores=stores)
    variants = {
        variant: builder.build(
            DEVELOPMENT_SNAPSHOTS,
            feature_window=variant,
        )
        for variant in ("full", "obs6")
    }

    keys = ["snapshot", "StoreID", "churn"]
    full_keys = variants["full"][keys].sort_values(keys).reset_index(drop=True)
    obs6_keys = variants["obs6"][keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(full_keys, obs6_keys)

    print("=== Friend Method — Corrected Development Audit ===", flush=True)
    print("Original notebooks unchanged.", flush=True)
    print("Historical-only safe features.", flush=True)
    print("Feature selection fitted separately in each training fold.", flush=True)
    print("Final-test labels NOT accessed.", flush=True)
    print()

    print("=== Verified Development Snapshots ===")
    print(
        variants["full"]
        .groupby("snapshot")
        .agg(
            observations=("StoreID", "size"),
            churners=("churn", "sum"),
        )
        .to_string()
    )
    print()

    metrics = []
    predictions = []

    for variant, frame in variants.items():
        for model_name in ("RandomForest", "LogReg"):
            for (
                fold_name,
                train_snapshots,
                validation_snapshot,
            ) in TEMPORAL_FOLDS:
                print(
                    f"Evaluating {variant} / {model_name} / {fold_name}",
                    flush=True,
                )

                fold_metrics, fold_predictions = run_fold(
                    frame=frame,
                    variant=variant,
                    model_name=model_name,
                    fold_name=fold_name,
                    train_snapshots=train_snapshots,
                    validation_snapshot=validation_snapshot,
                )
                metrics.append(fold_metrics)
                predictions.append(fold_predictions)

    metrics_frame = pd.DataFrame(metrics)
    predictions_frame = pd.concat(
        predictions, ignore_index=True
    )

    summary = (
        metrics_frame.groupby(["variant", "model"])
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            worst_fold_pr_auc=("pr_auc", "min"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1_at_0_5=("f1_at_0_5", "mean"),
        )
        .reset_index()
        .sort_values("mean_pr_auc", ascending=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "01_development_fold_metrics.csv": metrics_frame,
        "01_development_model_summary.csv": summary,
        "01_development_predictions.csv": predictions_frame,
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    print()
    print("=== Corrected Chronological Results ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("PR-AUC is the primary comparison metric.")
    print("F1 at threshold 0.5 is DIAGNOSTIC ONLY; no threshold is locked.")
    print("Final-test labels were NOT used.")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

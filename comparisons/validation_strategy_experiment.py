"""Controlled retrospective comparison of grouped and temporal selection.

The experiment deliberately keeps the protected 2013-10-01 snapshot out of
scope. Candidate configurations are selected using 2012-04-01 and 2012-10-01,
then evaluated on the same 2013-04-01 development snapshot.

The first selection snapshot has incomplete source coverage for its nominal
12-month feature window. Results are therefore retrospective evidence under a
relaxed history-coverage assumption, not a new final-test result.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.model_settings import (
    BASELINE_GRADIENT_BOOSTING,
    BASELINE_RANDOM_FOREST,
    MODEL_SEEDS,
)
from reseller_churn.config.project_paths import RESULTS_DIR
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.features.feature_pipeline import FeaturePipeline
from reseller_churn.modeling.model_factory import ModelFactory
from reseller_churn.modeling.resampling import (
    balance_classes_by_random_oversampling,
)


SELECTION_SNAPSHOTS = ("2012-04-01", "2012-10-01")
COMMON_EVALUATION_SNAPSHOT = "2013-04-01"
PROTECTED_TEST_SNAPSHOT = "2013-10-01"
ALERT_BUDGETS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPETITIONS = 5000
OUTPUT_DIR = RESULTS_DIR / "validation_strategy_experiment"
# Preserve this experiment's registered feature sets, which omit store_age.
# See docs/data_analysis.md for its historical-availability assumption.
# Adding it here would change the candidate search and require new evaluations.
EXPERIMENT_FEATURE_SETS = {
    name: [feature for feature in features if feature != "store_age"]
    for name, features in FEATURE_SETS.items()
}


@dataclass(frozen=True)
class Candidate:
    architecture: str
    feature_set: str
    imbalance: str

    @property
    def name(self) -> str:
        return f"{self.architecture}__{self.feature_set}__{self.imbalance}"

    @property
    def complexity_rank(self) -> tuple[int, int, int]:
        return (
            {"logistic": 0, "random_forest": 1, "gradient_boosting": 2}[
                self.architecture
            ],
            {"rfm": 0, "pruned": 1, "reduced": 2, "full": 3}[
                self.feature_set
            ],
            {"none": 0, "balanced_weights": 1, "random_oversampling": 2}[
                self.imbalance
            ],
        )


def candidate_manifest() -> list[Candidate]:
    """Return the identical, fixed candidate set used by both strategies."""
    return [
        Candidate(architecture, feature_set, imbalance)
        for architecture in (
            "logistic",
            "random_forest",
            "gradient_boosting",
        )
        for feature_set in FEATURE_SETS
        for imbalance in (
            "none",
            "balanced_weights",
            "random_oversampling",
        )
    ]


def build_labeled_snapshots(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    snapshots: tuple[str, ...],
) -> pd.DataFrame:
    """Build explicitly requested development snapshots only."""
    if PROTECTED_TEST_SNAPSHOT in snapshots:
        raise ValueError("Protected final-test snapshot is forbidden")

    for snapshot in snapshots:
        if PredictionWindow.from_date(snapshot).label_end > pd.Timestamp(PROTECTED_TEST_SNAPSHOT):
            raise ValueError(f"{snapshot}: labels cross the protected test date")

    features = FeaturePipeline().build(orders, stores, snapshots)
    labeled_parts = []

    for snapshot in snapshots:
        window = PredictionWindow.from_date(snapshot)
        if window.label_end > pd.Timestamp(PROTECTED_TEST_SNAPSHOT):
            raise ValueError(f"{snapshot}: labels cross the protected test date")

        snapshot_features = features[
            features["snapshot"] == window.snapshot
        ].copy()
        labels = build_churn_labels(orders, window)
        labeled = snapshot_features.merge(
            labels[["StoreID", "churn"]],
            on="StoreID",
            how="left",
            validate="one_to_one",
        )
        if labeled["churn"].isna().any():
            raise AssertionError(f"{snapshot}: missing churn label")
        labeled["churn"] = labeled["churn"].astype(int)
        labeled_parts.append(labeled)

    result = pd.concat(labeled_parts, ignore_index=True)
    if result.duplicated(["StoreID", "snapshot"]).any():
        raise AssertionError("Duplicate reseller/snapshot keys")
    return result


def _create_model(candidate: Candidate, seed: int):
    if candidate.architecture == "logistic":
        return ModelFactory.create_logistic_regression()
    if candidate.architecture == "random_forest":
        return ModelFactory.create_random_forest(
            BASELINE_RANDOM_FOREST,
            seed,
        )
    if candidate.architecture == "gradient_boosting":
        return ModelFactory.create_gradient_boosting(
            BASELINE_GRADIENT_BOOSTING,
            seed,
        )
    raise ValueError(f"Unknown architecture: {candidate.architecture}")


def fit_predict_candidate(
    candidate: Candidate,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    return_by_seed: bool = False,
) -> np.ndarray:
    """Fit preprocessing and imbalance handling inside one training fold."""
    features = EXPERIMENT_FEATURE_SETS[candidate.feature_set]
    x_train = training[features]
    y_train = training["churn"].astype(int)
    x_validation = validation[features]
    probabilities = []

    seeds = MODEL_SEEDS
    if candidate.architecture == "logistic" and candidate.imbalance != "random_oversampling":
        seeds = (MODEL_SEEDS[0],)

    for seed in seeds:
        current_x = x_train
        current_y = y_train
        fit_kwargs = {}

        if candidate.imbalance == "random_oversampling":
            current_x, current_y = balance_classes_by_random_oversampling(
                features=x_train,
                target=y_train,
                random_seed=seed,
            )
        elif candidate.imbalance == "balanced_weights":
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            if candidate.architecture == "gradient_boosting":
                fit_kwargs["sample_weight"] = weights
            else:
                fit_kwargs["model__sample_weight"] = weights
        elif candidate.imbalance != "none":
            raise ValueError(f"Unknown imbalance treatment: {candidate.imbalance}")

        model = _create_model(candidate, seed)
        model.fit(current_x, current_y, **fit_kwargs)
        probabilities.append(model.predict_proba(x_validation)[:, 1])

    by_seed = np.asarray(probabilities)
    # Deterministic logistic fits are identical across the five registered seeds.
    if len(by_seed) == 1:
        by_seed = np.repeat(by_seed, len(MODEL_SEEDS), axis=0)
    return by_seed if return_by_seed else np.mean(by_seed, axis=0)


def metrics_at_budget(
    target,
    probability,
    budget: float,
    tie_breaker=None,
) -> dict[str, float | int]:
    """Evaluate an exact top-fraction alert budget with deterministic ties."""
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not 0 < budget < 1:
        raise ValueError("Alert budget must be between zero and one")
    if len(target) != len(probability) or not len(target):
        raise ValueError("Target and probability must have equal nonzero length")

    if tie_breaker is None:
        tie_breaker = np.arange(len(target))
    tie_breaker = np.asarray(tie_breaker)
    alert_count = int(np.ceil(budget * len(target)))
    order = np.lexsort((tie_breaker, -probability))
    predicted = np.zeros(len(target), dtype=int)
    predicted[order[:alert_count]] = 1

    true_positives = int(np.sum((target == 1) & (predicted == 1)))
    false_positives = int(np.sum((target == 0) & (predicted == 1)))
    positives = int(target.sum())

    return {
        "budget": budget,
        "alert_count": alert_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": true_positives / positives if positives else 0.0,
        "precision": true_positives / alert_count if alert_count else 0.0,
        "f1": 2 * true_positives / (alert_count + positives) if alert_count + positives else 0.0,
    }


def _selection_folds(
    data: pd.DataFrame,
    strategy: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    working = data.reset_index(drop=True)
    if set(working["snapshot"].dt.strftime("%Y-%m-%d")) != set(SELECTION_SNAPSHOTS):
        raise AssertionError("Selection data contains unexpected snapshots")

    if strategy == "temporal":
        training = np.flatnonzero(
            working["snapshot"].eq(pd.Timestamp(SELECTION_SNAPSHOTS[0]))
        )
        validation = np.flatnonzero(
            working["snapshot"].eq(pd.Timestamp(SELECTION_SNAPSHOTS[1]))
        )
        if PredictionWindow.from_date(SELECTION_SNAPSHOTS[0]).label_end > pd.Timestamp(
            SELECTION_SNAPSHOTS[1]
        ):
            raise AssertionError("Temporal training labels are unavailable")
        return [("temporal_2012_10", training, validation)]

    if strategy == "group_kfold":
        folds = []
        splitter = GroupKFold(n_splits=5)
        for number, (training, validation) in enumerate(
            splitter.split(working, groups=working["StoreID"]),
            start=1,
        ):
            training_stores = set(working.iloc[training]["StoreID"])
            validation_stores = set(working.iloc[validation]["StoreID"])
            if training_stores & validation_stores:
                raise AssertionError("GroupKFold reused a reseller across split sides")
            folds.append((f"group_fold_{number}", training, validation))
        return folds

    raise ValueError(f"Unknown validation strategy: {strategy}")


def select_candidate(
    data: pd.DataFrame,
    strategy: str,
    candidates: list[Candidate],
) -> tuple[Candidate, pd.DataFrame, pd.DataFrame]:
    """Select a configuration without access to common-evaluation labels."""
    working = data.reset_index(drop=True)
    folds = _selection_folds(working, strategy)
    summary_rows = []
    fold_rows = []
    seed_rows = []

    for candidate_number, candidate in enumerate(candidates, start=1):
        print(
            f"  {strategy}: candidate {candidate_number:02d}/{len(candidates)} "
            f"{candidate.name}",
            flush=True,
        )
        fold_budget_recalls = []
        fold_pr_aucs = []

        for fold_name, training_indices, validation_indices in folds:
            training = working.iloc[training_indices]
            validation = working.iloc[validation_indices]
            by_seed = fit_predict_candidate(candidate, training, validation, return_by_seed=True)
            probability = np.mean(by_seed, axis=0)
            target = validation["churn"].to_numpy()
            for seed, seed_probability in zip(MODEL_SEEDS, by_seed):
                seed_rows.append({
                    "strategy": strategy,
                    "candidate": candidate.name,
                    "seed": seed,
                    "fold": fold_name,
                    "mean_budget_recall": float(np.mean([
                        metrics_at_budget(target, seed_probability, budget, validation["StoreID"])["recall"]
                        for budget in ALERT_BUDGETS
                    ])),
                    "pr_auc": average_precision_score(target, seed_probability),
                })
            pr_auc = average_precision_score(target, probability)
            fold_pr_aucs.append(pr_auc)

            row = {
                "strategy": strategy,
                "candidate": candidate.name,
                "fold": fold_name,
                "train_rows": len(training),
                "validation_rows": len(validation),
                "validation_churners": int(target.sum()),
                "pr_auc": pr_auc,
            }
            recalls = []
            for budget in ALERT_BUDGETS:
                budget_metrics = metrics_at_budget(
                    target,
                    probability,
                    budget,
                    tie_breaker=validation["StoreID"].to_numpy(),
                )
                label = int(round(budget * 100))
                row[f"recall_at_{label}"] = budget_metrics["recall"]
                row[f"precision_at_{label}"] = budget_metrics["precision"]
                row[f"false_positives_at_{label}"] = budget_metrics["false_positives"]
                recalls.append(float(budget_metrics["recall"]))

            row["mean_budget_recall"] = float(np.mean(recalls))
            fold_budget_recalls.append(row["mean_budget_recall"])
            fold_rows.append(row)

        summary_rows.append(
            {
                "strategy": strategy,
                "candidate": candidate.name,
                "architecture": candidate.architecture,
                "feature_set": candidate.feature_set,
                "imbalance": candidate.imbalance,
                "mean_budget_recall": float(np.mean(fold_budget_recalls)),
                "worst_fold_budget_recall": float(np.min(fold_budget_recalls)),
                "mean_pr_auc": float(np.mean(fold_pr_aucs)),
                "worst_fold_pr_auc": float(np.min(fold_pr_aucs)),
                "complexity_rank": candidate.complexity_rank,
            }
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).drop(columns="complexity_rank").to_csv(
            OUTPUT_DIR / f"{strategy}_selection_checkpoint.csv", index=False,
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["mean_budget_recall", "mean_pr_auc", "complexity_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected_name = summary.iloc[0]["candidate"]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / f"{strategy}_selection_seed_metrics.csv", index=False)
    return selected, summary, pd.DataFrame(fold_rows)


def bootstrap_recall_difference(
    target,
    group_probability,
    temporal_probability,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    random_seed: int = 2026,
    tie_breaker=None,
) -> tuple[float, float]:
    """Paired percentile interval for grouped minus temporal budget recall."""
    target = np.asarray(target, dtype=int)
    group_probability = np.asarray(group_probability, dtype=float)
    temporal_probability = np.asarray(temporal_probability, dtype=float)
    ties = np.arange(len(target)) if tie_breaker is None else np.asarray(tie_breaker)
    generator = np.random.default_rng(random_seed)
    differences = []

    for _ in range(repetitions):
        sample = generator.integers(0, len(target), size=len(target))
        sampled_target = target[sample]
        if sampled_target.sum() == 0:
            continue
        group_recall = np.mean(
            [
                metrics_at_budget(sampled_target, group_probability[sample], budget, ties[sample])[
                    "recall"
                ]
                for budget in ALERT_BUDGETS
            ]
        )
        temporal_recall = np.mean(
            [
                metrics_at_budget(
                    sampled_target,
                    temporal_probability[sample],
                    budget,
                    ties[sample],
                )["recall"]
                for budget in ALERT_BUDGETS
            ]
        )
        differences.append(group_recall - temporal_recall)

    return tuple(np.percentile(differences, [2.5, 97.5]))


def determine_winner(
    evaluation: pd.DataFrame,
    difference_interval: tuple[float, float],
) -> str:
    """Apply the predeclared conservative superiority rule."""
    pivot = evaluation.pivot(index="budget", columns="strategy", values="recall")
    differences = pivot["group_kfold"] - pivot["temporal"]
    group_wins = int((differences > 0).sum())
    temporal_wins = int((differences < 0).sum())

    if group_wins >= 2 and (differences >= 0).all() and difference_interval[0] > 0:
        return "group_kfold"
    if temporal_wins >= 2 and (differences <= 0).all() and difference_interval[1] < 0:
        return "temporal"
    return "inconclusive"


def main() -> None:
    print("=== Controlled Validation-Strategy Experiment ===")
    print("Selection: 2012-04-01 and 2012-10-01")
    print("Common development evaluation: 2013-04-01")
    print("Protected final test: NOT ACCESSED")
    print("History note: 2012-04-01 has incomplete 12-month source coverage.")

    orders = DatasetLoader.load_orders()
    # No transaction in the protected outcome period can enter this experiment.
    orders = orders[orders["OrderDate"] < pd.Timestamp(PROTECTED_TEST_SNAPSHOT)].copy()
    stores = DatasetLoader.load_stores()
    selection_data = build_labeled_snapshots(orders, stores, SELECTION_SNAPSHOTS)
    expected = {
        "2012-04-01": (200, 45),
        "2012-10-01": (366, 47),
    }
    for snapshot, (rows, churners) in expected.items():
        part = selection_data[
            selection_data["snapshot"] == pd.Timestamp(snapshot)
        ]
        if (len(part), int(part["churn"].sum())) != (rows, churners):
            raise AssertionError(f"{snapshot}: unexpected selection data")

    candidates = candidate_manifest()
    selected = {}
    summaries = []
    fold_metrics = []

    for strategy in ("group_kfold", "temporal"):
        print(f"\nSelecting with {strategy}...", flush=True)
        choice, summary, folds = select_candidate(
            selection_data,
            strategy,
            candidates,
        )
        selected[strategy] = choice
        summaries.append(summary)
        fold_metrics.append(folds)
        print(f"Selected: {choice.name}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selected": {strategy: asdict(candidate) for strategy, candidate in selected.items()},
        "seeds": list(MODEL_SEEDS),
        "budgets": list(ALERT_BUDGETS),
        "feature_sets": EXPERIMENT_FEATURE_SETS,
        "history_coverage": "partial at 2012-04-01",
    }
    (OUTPUT_DIR / "frozen_selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate common-evaluation predictions before constructing its labels.
    holdout_features = FeaturePipeline().build(
        orders,
        stores,
        (COMMON_EVALUATION_SNAPSHOT,),
    )
    if "churn" in holdout_features.columns:
        raise AssertionError("Common-evaluation features unexpectedly contain labels")

    holdout_by_seed = {
        strategy: fit_predict_candidate(candidate, selection_data, holdout_features, return_by_seed=True)
        for strategy, candidate in selected.items()
    }
    holdout_probability = {strategy: np.mean(scores, axis=0) for strategy, scores in holdout_by_seed.items()}
    frozen_predictions = holdout_features[["StoreID", "snapshot"]].copy()
    for strategy, probability in holdout_probability.items():
        frozen_predictions[f"{strategy}_probability"] = probability
    frozen_predictions.to_csv(OUTPUT_DIR / "frozen_unlabeled_predictions.csv", index=False)

    # Only now reveal the common development labels.
    holdout_labels = build_churn_labels(
        orders,
        PredictionWindow.from_date(COMMON_EVALUATION_SNAPSHOT),
    )
    holdout = holdout_features.merge(
        holdout_labels[["StoreID", "churn"]],
        on="StoreID",
        how="left",
        validate="one_to_one",
    )
    if (len(holdout), int(holdout["churn"].sum())) != (340, 64):
        raise AssertionError("Unexpected common-evaluation population")

    evaluation_rows = []
    seed_evaluation_rows = []
    prediction_output = holdout[["StoreID", "snapshot", "churn"]].copy()
    for strategy, probability in holdout_probability.items():
        for seed, scores in zip(MODEL_SEEDS, holdout_by_seed[strategy]):
            for budget in ALERT_BUDGETS:
                seed_evaluation_rows.append({
                    "strategy": strategy,
                    "seed": seed,
                    "pr_auc": average_precision_score(holdout["churn"], scores),
                    **metrics_at_budget(holdout["churn"], scores, budget, holdout["StoreID"]),
                })
        prediction_output[f"{strategy}_probability"] = probability
        pr_auc = average_precision_score(holdout["churn"], probability)
        for budget in ALERT_BUDGETS:
            evaluation_rows.append(
                {
                    "strategy": strategy,
                    "selected_candidate": selected[strategy].name,
                    "pr_auc": pr_auc,
                    **metrics_at_budget(
                        holdout["churn"],
                        probability,
                        budget,
                        tie_breaker=holdout["StoreID"],
                    ),
                }
            )

    evaluation = pd.DataFrame(evaluation_rows)
    interval = bootstrap_recall_difference(
        holdout["churn"],
        holdout_probability["group_kfold"],
        holdout_probability["temporal"],
        tie_breaker=holdout["StoreID"],
    )
    winner = determine_winner(evaluation, interval)
    seed_evaluation = pd.DataFrame(seed_evaluation_rows)
    seed_means = seed_evaluation.groupby(["seed", "strategy"])["recall"].mean().unstack()
    seed_differences = seed_means["group_kfold"] - seed_means["temporal"]
    if winner == "group_kfold" and (seed_differences < 0).any():
        winner = "inconclusive"
    if winner == "temporal" and (seed_differences > 0).any():
        winner = "inconclusive"
    seed_evaluation.to_csv(OUTPUT_DIR / "common_evaluation_seed_metrics.csv", index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).drop(columns="complexity_rank").to_csv(
        OUTPUT_DIR / "candidate_selection_summary.csv",
        index=False,
    )
    pd.concat(fold_metrics, ignore_index=True).to_csv(
        OUTPUT_DIR / "candidate_fold_metrics.csv",
        index=False,
    )
    prediction_output.to_csv(OUTPUT_DIR / "common_evaluation_predictions.csv", index=False)
    evaluation.to_csv(OUTPUT_DIR / "common_evaluation_metrics.csv", index=False)

    report = {
        "selection_snapshots": list(SELECTION_SNAPSHOTS),
        "common_evaluation_snapshot": COMMON_EVALUATION_SNAPSHOT,
        "protected_test_accessed": False,
        "candidate_count": len(candidates),
        "selected": {
            strategy: asdict(candidate) for strategy, candidate in selected.items()
        },
        "group_minus_temporal_mean_budget_recall_ci_95": list(interval),
        "winner": winner,
        "seed_mean_recall_differences": seed_differences.to_dict(),
        "interpretation": "retrospective_partial_history_development_comparison",
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Selected configurations ===")
    for strategy, candidate in selected.items():
        print(f"{strategy:12s}: {candidate.name}")

    print("\n=== Common April 2013 evaluation ===")
    display = evaluation[
        [
            "strategy",
            "budget",
            "alert_count",
            "true_positives",
            "false_positives",
            "recall",
            "precision",
            "pr_auc",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\n95% paired bootstrap CI, GroupKFold minus Temporal "
        f"mean budget recall: [{interval[0]:.4f}, {interval[1]:.4f}]"
    )
    print(f"Winner under predeclared rule: {winner}")
    print("Protected final-test labels were NOT accessed.")
    print(f"Saved results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

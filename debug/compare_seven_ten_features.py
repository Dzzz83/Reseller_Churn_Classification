"""Fixed seven-versus-ten comparison under temporal and grouped validation."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_companyx_features import FEATURE_GROUPS
from compare_rfm_seven_features import operating_metrics
from evaluate_companyx_features import paired_changes

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.model_settings import (
    BASELINE_GRADIENT_BOOSTING,
    BASELINE_RANDOM_FOREST,
    MODEL_SEEDS,
)
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.evaluation.selection import (
    SELECTION_RULE,
    THRESHOLDS,
    evaluate_threshold_tradeoffs,
    select_for_precision_floor,
)
from reseller_churn.modeling.development import (
    repository_identity,
    validate_labeled_development_data,
)
from reseller_churn.modeling.model_factory import ModelFactory

SOURCE = ROOT / "results/debugging/companyx_features"
OUTPUT = ROOT / "results/debugging/seven_vs_ten_features"
FEATURE_SETS = {
    "seven": list(PRUNED_FEATURES),
    "ten": list(PRUNED_FEATURES) + FEATURE_GROUPS["discounts"],
}
ARCHITECTURES = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def validation_splits(data):
    """Freeze shared splits; only the temporal splits enforce label availability."""
    temporal = TemporalDataset(data)
    splits = {"temporal": [], "grouped_kfold": []}
    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal.prepare_fold(fold)
        splits["temporal"].append((
            fold.name, prepared.training_data, prepared.validation_data,
        ))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for index, (train, validation) in enumerate(
        splitter.split(data, data["churn"], groups=data["StoreID"]), start=1
    ):
        training = data.iloc[train].copy()
        held_out = data.iloc[validation].copy()
        if set(training["StoreID"]) & set(held_out["StoreID"]):
            raise ValueError("Grouped fold reused a reseller across split sides.")
        if training["churn"].nunique() != 2 or held_out["churn"].nunique() != 2:
            raise ValueError("Grouped fold lacks both classes.")
        splits["grouped_kfold"].append((f"fold_{index}", training, held_out))
    validation_keys = pd.concat([
        held_out[["StoreID", "snapshot"]]
        for _, _, held_out in splits["grouped_kfold"]
    ])
    if len(validation_keys) != len(data) or validation_keys.duplicated().any():
        raise ValueError("Grouped validation does not cover every row exactly once.")
    return splits


def fit_scores(architecture, columns, training, validation, seed):
    """Use established architecture-specific preprocessing and training weights."""
    if architecture == "logistic_regression":
        model = ModelFactory.create_logistic_regression(c=1.0, max_iter=2000)
    elif architecture == "random_forest":
        model = ModelFactory.create_random_forest(BASELINE_RANDOM_FOREST, seed)
    elif architecture == "hist_gradient_boosting":
        model = ModelFactory.create_gradient_boosting(BASELINE_GRADIENT_BOOSTING, seed)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    weight_key = "sample_weight" if architecture == "hist_gradient_boosting" else "model__sample_weight"
    weights = compute_sample_weight("balanced", training["churn"])
    model.fit(training[columns], training["churn"], **{weight_key: weights})
    return model.predict_proba(validation[columns])[:, 1]


def summarize(predictions, method, architecture, score_kind, seed):
    """One common threshold per feature set, requiring the floor in every fold."""
    selections, tradeoffs = {}, []
    for name, cases in predictions.groupby("feature_set"):
        folds = {
            fold: {"target": group["actual_churn"].to_numpy(),
                   "probabilities": group["churn_probability"].to_numpy()}
            for fold, group in cases.groupby("fold")
        }
        table = evaluate_threshold_tradeoffs(f"{architecture}__{name}", folds)
        table["feature_set"] = name
        tradeoffs.append(table)
        selections[name] = select_for_precision_floor(table, .30)
    summary = pd.DataFrame([{"feature_set": name, **value} for name, value in selections.items()])
    metrics = operating_metrics(predictions, selections)
    tables = [summary, metrics, pd.concat(tradeoffs, ignore_index=True)]
    for table in tables:
        table["validation_method"] = method
        table["architecture"] = architecture
        table["score_kind"] = score_kind
        table["seed"] = seed
    return selections, tables


def compare_cases(predictions, selections):
    """Reuse matched-label checks; an infeasible pair has no operating comparison."""
    if any(value["status"] != "selected" for value in selections.values()):
        return pd.DataFrame()
    renamed = predictions.copy()
    renamed["feature_set"] = renamed["feature_set"].replace({"seven": "baseline", "ten": "discounts"})
    return paired_changes(
        renamed, {"baseline": selections["seven"], "discounts": selections["ten"]}, "discounts"
    ).drop(columns="feature_group")


def check_logistic_baseline(predictions):
    previous = pd.read_csv(SOURCE / "validation_predictions.csv")
    previous = previous[previous["feature_set"].isin(["baseline", "discounts"])].copy()
    previous["feature_set"] = previous["feature_set"].replace({"baseline": "seven", "discounts": "ten"})
    keys = ["fold", "StoreID", "snapshot", "feature_set"]
    current = predictions[
        predictions.validation_method.eq("temporal")
        & predictions.architecture.eq("logistic_regression")
    ]
    paired = current.merge(
        previous[keys + ["actual_churn", "churn_probability"]], on=keys,
        how="outer", validate="one_to_one", suffixes=("", "_previous"), indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Previous Logistic Regression population differs.")
    np.testing.assert_array_equal(paired.actual_churn, paired.actual_churn_previous)
    np.testing.assert_allclose(paired.churn_probability, paired.churn_probability_previous, atol=1e-10, rtol=0)
    return float((paired.churn_probability - paired.churn_probability_previous).abs().max())


def feature_deltas(summary):
    """Match configurations; do not calculate benefits for infeasible pairs."""
    summary = summary.copy()
    delta_metrics = ["mean_recall", "worst_fold_recall", "mean_precision", "pooled_tp", "pooled_fp"]
    for metric in delta_metrics:
        if metric not in summary:
            summary[metric] = np.nan
    keys = ["validation_method", "architecture", "score_kind", "seed"]
    paired = summary[summary.feature_set.eq("seven")].merge(
        summary[summary.feature_set.eq("ten")], on=keys, how="outer",
        validate="one_to_one", suffixes=("_seven", "_ten"), indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Summary configurations differ between feature sets.")
    paired["both_feasible"] = paired.status_seven.eq("selected") & paired.status_ten.eq("selected")
    for metric in delta_metrics:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_ten"] - paired[f"{metric}_seven"]
        ).where(paired.both_feasible)
    return paired.drop(columns="_merge")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE / "experiment_manifest.json").read_text())
    data_path = SOURCE / "development_features.csv"
    if file_sha256(data_path) != source_manifest["input_sha256"][str(data_path.relative_to(ROOT))]:
        raise ValueError("Source feature data changed since the earlier experiment.")
    data = validate_labeled_development_data(pd.read_csv(data_path))
    data = data.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    splits = validation_splits(data)
    protocol = {
        "feature_sets": FEATURE_SETS, "precision_floor": .30,
        "selection_rule": SELECTION_RULE, "thresholds": list(THRESHOLDS),
        "grouped_splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "seeds": list(MODEL_SEEDS), "logistic_fits_once_per_fold": True,
        "parameters": {
            "logistic_regression": {"C": 1.0, "max_iter": 2000},
            "random_forest": asdict(BASELINE_RANDOM_FOREST),
            "hist_gradient_boosting": asdict(BASELINE_GRADIENT_BOOSTING),
        },
        "imbalance_strategy": "balanced training sample weights", "tuning": False,
        "seed_aggregation": "mean predicted probability",
        "primary_validation": "temporal", "planned_fits": 154,
        "availability_assumption": "Existing order-time feature assumption retained; grouped folds mix periods and do not enforce chronological label availability",
    }
    write_json(OUTPUT / "experiment_protocol.json", protocol)
    memberships, counts = [], []
    for method, folds in splits.items():
        for fold, training, validation in folds:
            for role, cases in [("training", training), ("validation", validation)]:
                members = cases[["StoreID", "snapshot", "churn"]].copy()
                members["validation_method"], members["fold"], members["role"] = method, fold, role
                memberships.append(members)
                counts.append({
                    "validation_method": method, "fold": fold, "role": role,
                    "rows": len(cases), "resellers": cases.StoreID.nunique(),
                    "churners": int(cases.churn.sum()), "churn_rate": cases.churn.mean(),
                })
    pd.concat(memberships, ignore_index=True).to_csv(OUTPUT / "fold_membership.csv", index=False)
    pd.DataFrame(counts).to_csv(OUTPUT / "fold_counts.csv", index=False)
    ensemble_predictions, individual_predictions = [], []
    summaries, metric_tables, tradeoff_tables, change_tables = [], [], [], []
    models_fitted = 0
    for method, folds in splits.items():
        for architecture in ARCHITECTURES:
            seeds = (42,) if architecture == "logistic_regression" else MODEL_SEEDS
            individual = []
            for feature_set, columns in FEATURE_SETS.items():
                for fold, training, validation in folds:
                    for seed in seeds:
                        scores = fit_scores(architecture, columns, training, validation, seed)
                        models_fitted += 1
                        predicted = validation[["StoreID", "snapshot", "n_orders_3m"]].copy()
                        predicted["actual_churn"] = validation["churn"].to_numpy()
                        predicted["churn_probability"] = scores
                        predicted["validation_method"], predicted["architecture"] = method, architecture
                        predicted["feature_set"], predicted["fold"], predicted["seed"] = feature_set, fold, seed
                        individual.append(predicted)
            individuals = pd.concat(individual, ignore_index=True)
            individual_predictions.append(individuals)
            keys = ["validation_method", "architecture", "feature_set", "fold", "StoreID", "snapshot", "actual_churn", "n_orders_3m"]
            ensemble = individuals.groupby(keys, as_index=False)["churn_probability"].mean()
            ensemble_predictions.append(ensemble)
            selections, tables = summarize(ensemble, method, architecture, "ensemble", "ensemble")
            for destination, table in zip([summaries, metric_tables, tradeoff_tables], tables):
                destination.append(table)
            paired = compare_cases(ensemble, selections)
            if not paired.empty:
                paired["validation_method"], paired["architecture"] = method, architecture
                change_tables.append(paired)
            for seed, seed_cases in individuals.groupby("seed"):
                _, tables = summarize(seed_cases, method, architecture, "individual_seed", str(seed))
                for destination, table in zip([summaries, metric_tables, tradeoff_tables], tables):
                    destination.append(table)
            print(f"Completed {method}: {architecture}; fits so far {models_fitted}", flush=True)
    predictions = pd.concat(ensemble_predictions, ignore_index=True)
    baseline_error = check_logistic_baseline(predictions)
    if models_fitted != protocol["planned_fits"]:
        raise ValueError("Fit count differs from the approved comparison.")
    tables = {
        "validation_predictions.csv": predictions,
        "individual_seed_predictions.csv": pd.concat(individual_predictions, ignore_index=True),
        "selection_summary.csv": pd.concat(summaries, ignore_index=True),
        "fold_and_subgroup_metrics.csv": pd.concat(metric_tables, ignore_index=True),
        "threshold_tradeoffs.csv": pd.concat(tradeoff_tables, ignore_index=True),
    }
    deltas = feature_deltas(tables["selection_summary.csv"])
    tables["ensemble_feature_deltas.csv"] = deltas[deltas.score_kind.eq("ensemble")]
    tables["seed_feature_deltas.csv"] = deltas[deltas.score_kind.eq("individual_seed")]
    if change_tables:
        paired = pd.concat(change_tables, ignore_index=True)
        tables["paired_cases.csv"] = paired
        tables["changed_cases.csv"] = paired[paired.change_type.ne("unchanged")]
        tables["prediction_change_counts.csv"] = paired.groupby(
            ["validation_method", "architecture", "fold", "change_type"]
        ).size().rename("accounts").reset_index()
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False)
    code_paths = [
        Path(__file__), ROOT / "debug/evaluate_companyx_features.py",
        ROOT / "debug/compare_rfm_seven_features.py", ROOT / "debug/build_companyx_features.py",
        ROOT / "reseller_churn/modeling/model_factory.py",
        ROOT / "reseller_churn/evaluation/selection.py", ROOT / "reseller_churn/evaluation/metrics.py",
        ROOT / "reseller_churn/data/temporal_dataset.py",
        ROOT / "reseller_churn/config/model_settings.py",
        ROOT / "reseller_churn/config/validation_settings.py",
        ROOT / "reseller_churn/config/feature_sets.py",
    ]
    write_json(OUTPUT / "experiment_manifest.json", {
        "protocol": protocol, "models_fitted": models_fitted,
        "protected_final_test_labels_accessed": False,
        "baseline_max_absolute_score_error": baseline_error,
        "repository": repository_identity(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "input_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [data_path, SOURCE / "experiment_manifest.json", SOURCE / "validation_predictions.csv"]
        },
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths},
        "output_sha256": {
            name: file_sha256(OUTPUT / name)
            for name in [*tables, "fold_membership.csv", "fold_counts.csv", "experiment_protocol.json"]
        },
        "limitations": [
            "Development labels previously inspected; threshold selection is exploratory",
            "Grouped folds mix periods; they are not a deployment simulation",
            "Methods evaluate different populations and cannot be compared as matched targets",
            "Native missing handling for boosting, training-only imputation for LR/RF",
            "Only fixed baseline parameters and balanced weights were tested",
        ],
    })
    summary = tables["selection_summary.csv"]
    print(summary[summary.score_kind.eq("ensemble")][[
        "validation_method", "architecture", "feature_set", "status", "threshold",
        "mean_recall", "mean_precision", "pooled_tp", "pooled_fp",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Compare matched revenue definitions using shared stratified reseller folds."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_seven_ten_features import FEATURE_SETS, fit_scores
from compare_rfm_seven_features import operating_metrics
from evaluate_companyx_features import paired_changes
from prepare_subtotal_totaldue import REVENUE_COLUMNS
from reseller_churn.config.model_settings import (
    BASELINE_GRADIENT_BOOSTING, BASELINE_RANDOM_FOREST, MODEL_SEEDS,
)
from reseller_churn.config.project_paths import get_run_directory
from reseller_churn.evaluation.selection import (
    SELECTION_RULE, THRESHOLDS, evaluate_threshold_tradeoffs, select_for_precision_floor,
)
from reseller_churn.modeling.development import (
    file_sha256, repository_identity, validate_labeled_development_data, write_json,
)

CONFIGURATIONS = {"random_forest": "seven", "hist_gradient_boosting": "ten"}
PREVIOUS = ROOT / "results/debugging/seven_vs_ten_features"
KEYS = ["fold", "StoreID", "snapshot"]


def checked_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Recorded artifact hash differs: {path}")


def load_pair(preparation):
    """Check prepared identities and exact unchanged-column parity before fitting."""
    manifest = json.loads((preparation / "preparation_manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        checked_hash(preparation / name, expected)
    datasets = {}
    for definition in ["subtotal", "totaldue"]:
        data = pd.read_csv(preparation / f"development_{definition}.csv", float_precision="round_trip")
        data = validate_labeled_development_data(data)
        datasets[definition] = data.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    baseline, alternative = datasets.values()
    if not baseline.columns.equals(alternative.columns):
        raise ValueError("Prepared dataset columns differ")
    unchanged = baseline.columns.difference(REVENUE_COLUMNS)
    pd.testing.assert_frame_equal(baseline[unchanged], alternative[unchanged], check_exact=True)
    if len(baseline) != 1375 or baseline.churn.sum() != 212 or baseline.StoreID.nunique() != 422:
        raise ValueError("Unexpected prepared development population")
    for data in datasets.values():
        if not np.isfinite(data[REVENUE_COLUMNS].to_numpy()).all():
            raise ValueError("Non-finite revenue features")
    return datasets


def grouped_folds(data):
    """Return shared positional splits with identity separation and full coverage."""
    if data[["StoreID", "snapshot"]].duplicated().any():
        raise ValueError("Duplicate account/snapshot keys")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds, held_out = [], []
    for number, (train, validation) in enumerate(
        splitter.split(data, data.churn, groups=data.StoreID), start=1
    ):
        training, cases = data.iloc[train], data.iloc[validation]
        if set(training.StoreID) & set(cases.StoreID):
            raise ValueError("Reseller identity overlap")
        if training.churn.nunique() != 2 or cases.churn.nunique() != 2:
            raise ValueError("Each split side requires both classes")
        folds.append((f"fold_{number}", train, validation))
        held_out.extend(validation)
    if sorted(held_out) != list(range(len(data))):
        raise ValueError("Validation must cover each development row exactly once")
    return folds


def fold_membership(data, folds):
    tables = []
    for fold, train, validation in folds:
        for role, indices in [("training", train), ("validation", validation)]:
            table = data.iloc[indices][["StoreID", "snapshot", "churn"]].copy()
            table["fold"], table["role"] = fold, role
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def average_seeds(individual):
    """Require complete seed coverage per case before averaging probabilities."""
    keys = ["architecture", "feature_set", *KEYS]
    if individual[keys + ["seed"]].duplicated().any():
        raise ValueError("Duplicate seed predictions")
    groups = individual.groupby(keys, dropna=False)
    if not groups.seed.apply(lambda values: set(values) == set(MODEL_SEEDS)).all():
        raise ValueError("Incomplete seed coverage")
    for column in ["actual_churn", "n_orders_3m"]:
        if not groups[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Seed cases disagree on {column}")
    return groups.agg(
        actual_churn=("actual_churn", "first"),
        n_orders_3m=("n_orders_3m", "first"),
        churn_probability=("churn_probability", "mean"),
    ).reset_index()


def metrics_at_cutoffs(predictions, selections):
    """Report every fold and pooled subgroups at the supplied ensemble cutoffs."""
    pooled = predictions.copy()
    pooled["fold"] = "pooled"
    return operating_metrics(pd.concat([predictions, pooled], ignore_index=True), selections)


def summarize(predictions):
    selections, tradeoffs = {}, []
    for definition, cases in predictions.groupby("feature_set"):
        folds = {
            fold: {"target": group.actual_churn.to_numpy(),
                   "probabilities": group.churn_probability.to_numpy()}
            for fold, group in cases.groupby("fold")
        }
        table = evaluate_threshold_tradeoffs(definition, folds)
        table["feature_set"] = definition
        tradeoffs.append(table)
        selections[definition] = select_for_precision_floor(table, .30)
    summary = pd.DataFrame([{"feature_set": name, **rule} for name, rule in selections.items()])
    return selections, summary, pd.concat(tradeoffs, ignore_index=True)


def matched_cases(predictions, selections):
    if any(rule["status"] != "selected" for rule in selections.values()):
        return pd.DataFrame()
    renamed = predictions.copy()
    renamed["feature_set"] = renamed.feature_set.replace({"subtotal": "baseline"})
    return paired_changes(
        renamed, {"baseline": selections["subtotal"], "totaldue": selections["totaldue"]}, "totaldue"
    )


def verify_baseline(predictions, previous):
    """Do not attribute a changed historical baseline to the new revenue variant."""
    current = predictions[predictions.feature_set.eq("subtotal")]
    saved = previous[previous.validation_method.eq("grouped_kfold")].copy()
    tables = []
    for architecture, feature_set in CONFIGURATIONS.items():
        actual = current[current.architecture.eq(architecture)]
        expected = saved[saved.architecture.eq(architecture) & saved.feature_set.eq(feature_set)]
        paired = actual.merge(expected[KEYS + ["actual_churn", "churn_probability"]],
                              on=KEYS, how="outer", validate="one_to_one",
                              suffixes=("", "_saved"), indicator=True)
        if not paired._merge.eq("both").all():
            raise ValueError("Historical baseline population differs")
        np.testing.assert_array_equal(paired.actual_churn, paired.actual_churn_saved)
        np.testing.assert_allclose(paired.churn_probability, paired.churn_probability_saved, atol=1e-10, rtol=0)
        tables.append({"architecture": architecture,
                       "max_absolute_score_error": float(abs(paired.churn_probability - paired.churn_probability_saved).max())})
    return tables


def run(preparation, run_id):
    get_run_directory(run_id)
    output = ROOT / "results/debugging/subtotal_vs_totaldue" / run_id
    if output.exists():
        raise FileExistsError(f"Choose a new run ID; directory exists: {output}")
    data = load_pair(preparation)
    baseline = data["subtotal"]
    folds = grouped_folds(baseline)
    membership = fold_membership(baseline, folds)
    old_manifest_path = PREVIOUS / "experiment_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    for name in ["fold_membership.csv", "validation_predictions.csv"]:
        checked_hash(PREVIOUS / name, old_manifest["output_sha256"][name])
    old_membership = pd.read_csv(PREVIOUS / "fold_membership.csv")
    old_membership = old_membership[old_membership.validation_method.eq("grouped_kfold")]
    columns = membership.columns.tolist()
    order = ["fold", "role", "snapshot", "StoreID"]
    pd.testing.assert_frame_equal(
        membership.sort_values(order).reset_index(drop=True),
        old_membership[columns].sort_values(order).reset_index(drop=True), check_exact=True,
    )
    protocol = {
        "validation": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "group": "StoreID", "feature_sets": {arch: FEATURE_SETS[name] for arch, name in CONFIGURATIONS.items()},
        "changed_columns": REVENUE_COLUMNS, "seeds": list(MODEL_SEEDS), "planned_fits": 100,
        "parameters": {"random_forest": asdict(BASELINE_RANDOM_FOREST),
                       "hist_gradient_boosting": asdict(BASELINE_GRADIENT_BOOSTING)},
        "early_stopping": False, "weights": "balanced training sample weights",
        "seed_aggregation": "mean predicted probability", "precision_floor": .30,
        "selection_rule": SELECTION_RULE, "thresholds": list(THRESHOLDS),
        "seed_diagnostic_cutoff": "corresponding ensemble selected cutoff",
        "additional_diagnostic": "both variants at SubTotal ensemble selected cutoff",
        "final_test_outcomes_accessed": False,
    }
    output.mkdir(parents=True)
    write_json(output / "experiment_protocol.json", protocol)
    (output / "protocol.md").write_text((preparation / "protocol.md").read_text())
    membership.to_csv(output / "fold_membership.csv", index=False)
    membership.groupby(["fold", "role"]).agg(
        rows=("StoreID", "size"), resellers=("StoreID", "nunique"), churners=("churn", "sum")
    ).reset_index().to_csv(output / "fold_counts.csv", index=False)
    individual, fit_count = [], 0
    for architecture, feature_set in CONFIGURATIONS.items():
        for definition, frame in data.items():
            for fold, train, validation in folds:
                training, cases = frame.iloc[train], frame.iloc[validation]
                for seed in MODEL_SEEDS:
                    probabilities = fit_scores(architecture, FEATURE_SETS[feature_set], training, cases, seed)
                    predicted = cases[["StoreID", "snapshot", "n_orders_3m"]].copy()
                    predicted["actual_churn"] = cases.churn.to_numpy()
                    predicted["churn_probability"] = probabilities
                    predicted["architecture"], predicted["feature_set"] = architecture, definition
                    predicted["fold"], predicted["seed"] = fold, seed
                    individual.append(predicted)
                    fit_count += 1
                print(f"Completed {architecture} {definition} {fold}; {fit_count}/100 fits", flush=True)
    individuals = pd.concat(individual, ignore_index=True)
    ensemble = average_seeds(individuals)
    baseline_checks = verify_baseline(ensemble, pd.read_csv(PREVIOUS / "validation_predictions.csv"))
    summaries, tradeoffs, metric_tables, seed_metrics, diagnostics, paired = [], [], [], [], [], []
    for architecture, cases in ensemble.groupby("architecture"):
        selections, summary, grid = summarize(cases)
        metrics = metrics_at_cutoffs(cases, selections)
        seed_tables = []
        for seed, group in individuals[individuals.architecture.eq(architecture)].groupby("seed"):
            table = metrics_at_cutoffs(group, selections)
            table["seed"] = seed
            seed_tables.append(table)
        changes = matched_cases(cases, selections)
        diagnostic = pd.DataFrame()
        if selections["subtotal"]["status"] == "selected":
            shared = {name: selections["subtotal"] for name in selections}
            diagnostic = metrics_at_cutoffs(cases, shared)
        for destination, table in [(summaries, summary), (tradeoffs, grid), (metric_tables, metrics),
                                   (seed_metrics, pd.concat(seed_tables, ignore_index=True)),
                                   (diagnostics, diagnostic), (paired, changes)]:
            table["architecture"] = architecture
            destination.append(table)
    tables = {
        "individual_seed_predictions.csv": individuals, "validation_predictions.csv": ensemble,
        "selection_summary.csv": pd.concat(summaries, ignore_index=True),
        "threshold_tradeoffs.csv": pd.concat(tradeoffs, ignore_index=True),
        "fold_and_subgroup_metrics.csv": pd.concat(metric_tables, ignore_index=True),
        "seed_metrics_at_ensemble_cutoff.csv": pd.concat(seed_metrics, ignore_index=True),
        "metrics_at_subtotal_cutoff.csv": pd.concat(diagnostics, ignore_index=True),
        "paired_cases.csv": pd.concat(paired, ignore_index=True),
    }
    changes = tables["paired_cases.csv"]
    if "change_type" in changes:
        tables["changed_cases.csv"] = changes[changes.change_type.ne("unchanged")]
        tables["prediction_change_counts.csv"] = changes.groupby(
            ["architecture", "fold", "change_type"]
        ).size().rename("cases").reset_index()
    if fit_count != 100:
        raise ValueError("Unexpected fit count")
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    code_paths = [Path(__file__), *[ROOT / path for path in [
        "debug/compare_seven_ten_features.py", "debug/compare_rfm_seven_features.py",
        "debug/evaluate_companyx_features.py", "debug/build_companyx_features.py",
        "debug/prepare_subtotal_totaldue.py", "reseller_churn/modeling/model_factory.py",
        "reseller_churn/config/model_settings.py", "reseller_churn/config/feature_sets.py",
        "reseller_churn/evaluation/selection.py", "reseller_churn/evaluation/metrics.py",
    ]]]
    inputs = [preparation / name for name in ["preparation_manifest.json", "protocol.md",
              "development_subtotal.csv", "development_totaldue.csv"]]
    inputs += [old_manifest_path, PREVIOUS / "fold_membership.csv", PREVIOUS / "validation_predictions.csv"]
    write_json(output / "experiment_manifest.json", {
        "protocol": protocol, "models_fitted": fit_count, "baseline_checks": baseline_checks,
        "final_test_outcomes_accessed": False, "repository": repository_identity(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in code_paths},
        "output_sha256": {name: file_sha256(output / name) for name in [
            *tables, "fold_membership.csv", "fold_counts.csv", "protocol.md", "experiment_protocol.json"]},
        "limitations": ["Inspected development labels and selected cutoffs: exploratory results",
                        "Grouped folds mix periods and do not enforce historical label availability"],
    })
    print(tables["selection_summary.csv"][["architecture", "feature_set", "status", "threshold",
          "mean_recall", "mean_precision", "pooled_tp", "pooled_fp"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    run(arguments.preparation.resolve(), arguments.run_id)

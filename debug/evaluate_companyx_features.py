"""Describe associations, then measure fixed feature groups against baseline."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_companyx_features import FEATURE_GROUPS, OUTPUT
from compare_rfm_seven_features import operating_metrics
from reseller_churn.config.feature_sets import PRUNED_FEATURES
from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.evaluation.selection import (
    SELECTION_RULE, THRESHOLDS, evaluate_threshold_tradeoffs, select_for_precision_floor,
)
from reseller_churn.modeling.development import (
    file_sha256, repository_identity, validate_labeled_development_data, write_json,
)
from reseller_churn.modeling.model_factory import ModelFactory


BASELINE_RUN = ROOT / "results/debugging/rfm_vs_seven_features_precision_030"
NEW_FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]


def benjamini_hochberg(values):
    """Adjust finite p-values within one period/subgroup/method family."""
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    result = np.full(len(values), np.nan)
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1)
    return result


def association_tables(data):
    """Each snapshot contains unique accounts; no pooled repeated-row p-values."""
    numeric_rows, categorical_rows = [], []
    for snapshot, period in data.groupby("snapshot"):
        for subgroup, cases in [
            ("overall", period),
            ("purchased_in_3m", period[period["n_orders_3m"].gt(0)]),
        ]:
            for name in NEW_FEATURES:
                observed = cases[[name, "churn"]].dropna()
                row = {
                    "snapshot": snapshot, "subgroup": subgroup, "feature": name,
                    "accounts": len(cases), "observed": len(observed),
                    "missing": int(cases[name].isna().sum()),
                    "unique_values": observed[name].nunique(),
                    "observed_churners": int(observed["churn"].sum()),
                    "missing_churners": int(cases.loc[cases[name].isna(), "churn"].sum()),
                    "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                }
                if len(observed) >= 3 and observed[name].nunique() > 1 and observed["churn"].nunique() > 1:
                    pearson = pearsonr(observed[name], observed["churn"])
                    spearman = spearmanr(observed[name], observed["churn"])
                    row.update(
                        pearson_r=float(pearson.statistic), pearson_p=float(pearson.pvalue),
                        spearman_r=float(spearman.statistic), spearman_p=float(spearman.pvalue),
                    )
                numeric_rows.append(row)
            for name in ["last_salesperson", "last_territory"]:
                rates = cases.groupby(name, dropna=False)["churn"].agg(["size", "sum", "mean"])
                for category, row in rates.iterrows():
                    categorical_rows.append({
                        "snapshot": snapshot, "subgroup": subgroup, "field": name,
                        "category": category, "accounts": int(row["size"]),
                        "churners": int(row["sum"]), "churn_rate": float(row["mean"]),
                    })
    numeric = pd.DataFrame(numeric_rows)
    for _, indices in numeric.groupby(["snapshot", "subgroup"]).groups.items():
        for method in ["pearson", "spearman"]:
            numeric.loc[indices, f"{method}_q"] = benjamini_hochberg(
                numeric.loc[indices, f"{method}_p"]
            )
    numeric["role"] = np.where(
        numeric["snapshot"].isin(["2012-07-01", "2012-10-01"]),
        "training_period", "inspected_validation_period",
    )
    training = data[data["snapshot"].isin(DEVELOPMENT_FOLDS[1].training_snapshots)]
    redundancy = []
    for new_feature in NEW_FEATURES:
        for existing_feature in PRUNED_FEATURES:
            pair = training[[new_feature, existing_feature]].dropna()
            coefficient = np.nan
            if pair[new_feature].nunique() > 1 and pair[existing_feature].nunique() > 1:
                coefficient = float(spearmanr(pair[new_feature], pair[existing_feature]).statistic)
            redundancy.append({
                "new_feature": new_feature, "existing_feature": existing_feature,
                "observed_training_rows": len(pair), "spearman_r": coefficient,
            })
    return numeric, pd.DataFrame(categorical_rows), pd.DataFrame(redundancy)


def paired_changes(predictions, selections, group_name):
    keys = ["fold", "StoreID", "snapshot"]
    baseline = predictions[predictions["feature_set"].eq("baseline")].copy()
    augmented = predictions[predictions["feature_set"].eq(group_name)].copy()
    baseline["baseline_alert"] = baseline["churn_probability"].ge(selections["baseline"]["threshold"]).astype(int)
    augmented["augmented_alert"] = augmented["churn_probability"].ge(selections[group_name]["threshold"]).astype(int)
    baseline = baseline.rename(columns={"churn_probability": "baseline_score"})
    augmented = augmented.rename(columns={"churn_probability": "augmented_score"})
    augmented = augmented.rename(columns={"actual_churn": "augmented_churn"})
    paired = baseline[keys + ["actual_churn", "n_orders_3m", "baseline_score", "baseline_alert"]].merge(
        augmented[keys + ["augmented_churn", "augmented_score", "augmented_alert"]],
        on=keys, validate="one_to_one", how="outer", indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Feature-group cases differ from baseline.")
    if not paired["actual_churn"].eq(paired["augmented_churn"]).all():
        raise ValueError("Feature-group labels differ from baseline.")
    positive = paired["actual_churn"].eq(1)
    gain = paired["baseline_alert"].eq(0) & paired["augmented_alert"].eq(1)
    lost = paired["baseline_alert"].eq(1) & paired["augmented_alert"].eq(0)
    paired["change_type"] = np.select(
        [positive & gain, positive & lost, ~positive & gain, ~positive & lost],
        ["newly_caught_churn", "newly_missed_churn", "added_false_alert", "removed_false_alert"],
        default="unchanged",
    )
    paired["feature_group"] = group_name
    return paired.drop(columns=["_merge", "augmented_churn"])


def main():
    protocol = json.loads((OUTPUT / "feature_protocol.json").read_text())
    if protocol["groups"] != FEATURE_GROUPS:
        raise ValueError("Feature groups differ from the frozen protocol.")
    data_path = OUTPUT / "development_features.csv"
    data = validate_labeled_development_data(pd.read_csv(data_path))
    numeric, categorical, redundancy = association_tables(data)
    tables = {
        "numeric_associations.csv": numeric,
        "categorical_churn_rates.csv": categorical,
        "training_feature_redundancy.csv": redundancy,
    }
    configurations = {"baseline": list(PRUNED_FEATURES)}
    configurations.update({name: list(PRUNED_FEATURES) + group for name, group in FEATURE_GROUPS.items()})
    temporal = TemporalDataset(data)
    all_predictions, all_tradeoffs = [], []
    selections = {}
    for name, columns in configurations.items():
        folds = {}
        for fold in DEVELOPMENT_FOLDS:
            prepared = temporal.prepare_fold(fold)
            training = prepared.training_data
            validation = prepared.validation_data
            model = ModelFactory.create_logistic_regression(c=1.0, max_iter=2000)
            weights = compute_sample_weight("balanced", training["churn"])
            model.fit(training[columns], training["churn"], model__sample_weight=weights)
            scores = model.predict_proba(validation[columns])[:, 1]
            folds[fold.name] = {"target": validation["churn"].to_numpy(), "probabilities": scores}
            predicted = validation.copy()
            predicted["feature_set"] = name
            predicted["fold"] = fold.name
            predicted["actual_churn"] = validation["churn"]
            predicted["churn_probability"] = scores
            all_predictions.append(predicted)
        tradeoffs = evaluate_threshold_tradeoffs(name, folds)
        tradeoffs["feature_group"] = name
        all_tradeoffs.append(tradeoffs)
        selections[name] = select_for_precision_floor(tradeoffs, .30)
    predictions = pd.concat(all_predictions, ignore_index=True)
    saved = pd.read_csv(BASELINE_RUN / "validation_predictions.csv")
    saved = saved[saved["feature_set"].eq("pruned")]
    baseline = predictions[predictions["feature_set"].eq("baseline")]
    baseline_check = baseline.merge(
        saved[["fold", "StoreID", "snapshot", "actual_churn", "churn_probability"]],
        on=["fold", "StoreID", "snapshot"], how="outer", validate="one_to_one",
        suffixes=("", "_saved"), indicator=True,
    )
    if not baseline_check["_merge"].eq("both").all():
        raise ValueError("Seven-feature baseline cases changed.")
    if not baseline_check["actual_churn"].eq(baseline_check["actual_churn_saved"]).all():
        raise ValueError("Seven-feature baseline labels changed.")
    np.testing.assert_allclose(
        baseline_check["churn_probability"], baseline_check["churn_probability_saved"],
        atol=1e-10, rtol=0,
    )
    saved_manifest = json.loads((BASELINE_RUN / "experiment_manifest.json").read_text())
    if selections["baseline"]["threshold"] != saved_manifest["selections"]["pruned"]["threshold"]:
        raise ValueError("Baseline operating point changed.")
    metrics = operating_metrics(predictions, selections)
    changes = []
    for name in FEATURE_GROUPS:
        if selections[name]["status"] == "selected":
            changes.append(paired_changes(predictions, selections, name))
    tables.update({
        "validation_predictions.csv": predictions,
        "fold_and_subgroup_metrics.csv": metrics,
        "threshold_tradeoffs.csv": pd.concat(all_tradeoffs, ignore_index=True),
        "selection_summary.csv": pd.DataFrame([
            {"feature_group": name, **value} for name, value in selections.items()
        ]),
    })
    if changes:
        paired = pd.concat(changes, ignore_index=True)
        tables["paired_cases.csv"] = paired
        tables["changed_cases.csv"] = paired[paired["change_type"].ne("unchanged")]
        tables["prediction_change_counts.csv"] = paired.groupby(
            ["feature_group", "fold", "change_type"]
        ).size().rename("accounts").reset_index()
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False)
    input_paths = [data_path, OUTPUT / "feature_protocol.json", BASELINE_RUN / "experiment_manifest.json", BASELINE_RUN / "validation_predictions.csv"]
    source_paths = sorted((OUTPUT / "source").glob("*.csv"))
    code_paths = [
        ROOT / "debug/companyx_readonly.py", ROOT / "debug/inventory_companyx.py",
        ROOT / "debug/extract_companyx_history.py", ROOT / "debug/build_companyx_features.py",
        Path(__file__), ROOT / "debug/compare_rfm_seven_features.py",
        ROOT / "reseller_churn/modeling/model_factory.py",
        ROOT / "reseller_churn/evaluation/selection.py",
        ROOT / "reseller_churn/evaluation/metrics.py",
    ]
    import scipy
    import sklearn
    write_json(OUTPUT / "experiment_manifest.json", {
        "configurations": configurations, "selections": selections,
        "model": "weighted logistic regression, C=1.0, max_iter=2000; original seed policy 42",
        "preprocessing": "Training-only median imputation with indicators and standard scaling",
        "selection_rule": SELECTION_RULE, "thresholds": list(THRESHOLDS),
        "precision_floor": .30, "models_fitted": 8, "tuning": False,
        "protected_final_test_labels_accessed": False,
        "baseline_max_absolute_score_error": float((baseline_check["churn_probability"] - baseline_check["churn_probability_saved"]).abs().max()),
        "correlation_protocol": "Pearson and Spearman per snapshot/subgroup; BH within each 11-feature method family; descriptive only, no feature selection based on correlations",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "repository": repository_identity(),
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in input_paths + source_paths},
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths},
        "output_sha256": {name: file_sha256(OUTPUT / name) for name in tables},
        "limitations": [
            "Already inspected development labels also select operating points",
            "Three separate group comparisons; no independent confirmatory estimate",
            "Order-line availability at order time is assumed, not verified from version history",
            "Changing profiles/status and constant shipping fields excluded",
        ],
    })
    print(metrics[metrics["subgroup"].eq("overall")].to_string(index=False))
    print(tables["selection_summary.csv"][[
        "feature_group", "status", "threshold", "mean_recall", "mean_precision",
        "worst_fold_recall", "pooled_tp", "pooled_fp",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

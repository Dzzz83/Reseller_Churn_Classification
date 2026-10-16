"""Compare frozen revenue variants on the already-inspected October cohort."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from companyx_readonly import query
from compare_seven_ten_features import FEATURE_SETS
from compare_subtotal_totaldue import CONFIGURATIONS, checked_hash, load_pair, metrics_at_cutoffs
from prepare_subtotal_totaldue import HEADER_COLUMNS, REVENUE_COLUMNS, build_alternative
from reseller_churn.config.model_settings import BASELINE_RANDOM_FOREST, BASELINE_GRADIENT_BOOSTING, MODEL_SEEDS
from reseller_churn.config.project_paths import get_run_directory
from reseller_churn.evaluation.selection import select_for_precision_floor
from reseller_churn.modeling.development import file_sha256, write_json
from reseller_churn.modeling.model_factory import ModelFactory

ORIGINAL = ROOT / "results/final_evaluation/gb10_grouped_20261011"
RF_ORIGINAL = ROOT / "results/final_evaluation/rf7_grouped_20261014"
SNAPSHOT = pd.Timestamp("2013-10-01")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_final_features(path):
    """Preserve serialized feature values at fitted histogram boundaries."""
    features = pd.read_csv(path, float_precision="round_trip").sort_values("StoreID").reset_index(drop=True)
    if "churn" in features or "actual_churn" in features:
        raise ValueError("Scoring features contain labels")
    if len(features) != 492 or features.StoreID.duplicated().any() or not features.snapshot.eq("2013-10-01").all():
        raise ValueError("Unexpected scoring cohort")
    return features


def final_headers(clean, raw):
    """Reconcile only accepted pre-October order amounts; exclude all outcomes."""
    raw = raw.copy()
    raw["OrderDate"] = pd.to_datetime(raw.OrderDate)
    if raw[HEADER_COLUMNS].isna().any().any() or raw.SalesOrderID.duplicated().any():
        raise ValueError("Missing or duplicate raw order facts")
    if raw.OrderDate.ge(SNAPSHOT).any():
        raise ValueError("Extract includes October or later orders")
    accepted = clean.loc[clean.OrderDate.lt(SNAPSHOT), HEADER_COLUMNS[:4]]
    paired = accepted.merge(raw, on="SalesOrderID", how="left", validate="one_to_one",
                            suffixes=("_clean", ""), indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Missing accepted orders")
    for name in ["StoreID", "OrderDate"]:
        if not paired[name].eq(paired[f"{name}_clean"]).all():
            raise ValueError(f"Source mismatch: {name}")
    amounts = paired[["SubTotal", "TaxAmt", "Freight", "TotalDue"]]
    if not np.isfinite(amounts.to_numpy()).all():
        raise ValueError("Non-finite amounts")
    if (amounts[["SubTotal", "TotalDue"]] <= 0).any().any() or (amounts[["TaxAmt", "Freight"]] < 0).any().any():
        raise ValueError("Invalid monetary amounts")
    if not np.allclose(paired.SubTotal, paired.SubTotal_clean, atol=1e-6, rtol=0):
        raise ValueError("SubTotal mismatch")
    if not np.allclose(paired.TotalDue, paired.SubTotal + paired.TaxAmt + paired.Freight, atol=1e-4, rtol=0):
        raise ValueError("TotalDue formula mismatch")
    return paired[HEADER_COLUMNS].sort_values("SalesOrderID").reset_index(drop=True)


def attach_labels(predictions, labels):
    """Require the same known cohort for every configuration."""
    keys = ["StoreID", "snapshot"]
    if labels[keys].duplicated().any() or not labels.actual_churn.isin([0, 1]).all():
        raise ValueError("Invalid outcome keys or values")
    tables = []
    for _, cases in predictions.groupby(["architecture", "feature_set"]):
        joined = cases.merge(labels, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all() or len(joined) != 492 or joined.actual_churn.sum() != 46:
            raise ValueError("Final outcome population differs")
        tables.append(joined.drop(columns="_merge"))
    return pd.concat(tables, ignore_index=True)


def run(preparation, comparison, run_id):
    get_run_directory(run_id)
    output = ROOT / "results/final_evaluation" / run_id
    model_directory = ROOT / "models/provisional" / run_id
    if output.exists() or model_directory.exists():
        raise FileExistsError("Choose a new run ID; evaluation/model directory exists")
    data = load_pair(preparation)
    comparison_manifest = json.loads((comparison / "experiment_manifest.json").read_text())
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in comparison_manifest[category].items():
            checked_hash(ROOT / name, expected)
    for name, expected in comparison_manifest["output_sha256"].items():
        checked_hash(comparison / name, expected)
    grid = pd.read_csv(comparison / "threshold_tradeoffs.csv", float_precision="round_trip")
    rules = {}
    for architecture in CONFIGURATIONS:
        rules[architecture] = {}
        for definition in data:
            subset = grid[grid.architecture.eq(architecture) & grid.feature_set.eq(definition)]
            selected = select_for_precision_floor(subset, .30)
            if selected["status"] != "selected":
                raise ValueError("Infeasible frozen development configuration")
            rules[architecture][definition] = selected
    checkpoint = json.loads((ORIGINAL / "prediction_checkpoint.json").read_text())
    feature_path = ORIGINAL / "final_test_features.csv"
    checked_hash(feature_path, checkpoint["output_sha256"][feature_path.name])
    baseline = load_final_features(feature_path)
    for frame in data.values():
        label_ends = pd.to_datetime(frame.snapshot) + pd.DateOffset(months=6)
        if label_ends.gt(SNAPSHOT).any():
            raise ValueError("Training labels unavailable at prediction snapshot")
    raw = query("""
SELECT h.SalesOrderID, c.StoreID, CONVERT(varchar(23),h.OrderDate,121) AS OrderDate,
       h.SubTotal, h.TaxAmt, h.Freight, h.TotalDue
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20131001'
ORDER BY h.SalesOrderID;
""")
    clean_path = ROOT / "datasets/orders_clean.csv"
    headers = final_headers(pd.read_csv(clean_path, parse_dates=["OrderDate"]), raw)
    features = {"subtotal": baseline, "totaldue": build_alternative(baseline, headers)}
    protocol = {
        "recorded_at_utc": now(), "authorization": "User: evaluate on the final test set too, 2026-10-15",
        "test_outcomes_previously_inspected": True, "independent_untouched_test": False,
        "test_snapshot": "2013-10-01", "label_end_exclusive": "2014-04-01",
        "rules": rules, "minimum_development_precision": .30,
        "methodology": comparison_manifest["protocol"], "planned_fits": 20,
        "training_rows": 1375, "scoring_rows": 492,
        "changed_features": REVENUE_COLUMNS,
        "label_policy": "Reuse existing inspected labels only after saving frozen scores; no new outcome extraction",
    }
    output.mkdir(parents=True)
    model_directory.mkdir(parents=True)
    write_json(output / "frozen_protocol.json", protocol)
    raw.to_csv(output / "historical_headers_raw.csv", index=False)
    headers.to_csv(output / "accepted_historical_headers.csv", index=False)
    for definition, frame in features.items():
        frame.to_csv(output / f"features_{definition}.csv", index=False)
    predictions, individuals, fitted = [], [], 0
    model_paths = []
    for architecture, feature_set in CONFIGURATIONS.items():
        columns = FEATURE_SETS[feature_set]
        for definition, training in data.items():
            scores, models = [], []
            weights = compute_sample_weight("balanced", training.churn)
            for seed in MODEL_SEEDS:
                if architecture == "random_forest":
                    model = ModelFactory.create_random_forest(BASELINE_RANDOM_FOREST, seed)
                    weight_key = "model__sample_weight"
                else:
                    model = ModelFactory.create_gradient_boosting(BASELINE_GRADIENT_BOOSTING, seed)
                    weight_key = "sample_weight"
                model.fit(training[columns], training.churn, **{weight_key: weights})
                score = model.predict_proba(features[definition][columns])[:, 1]
                models.append(model)
                scores.append(score)
                item = features[definition][["StoreID", "snapshot", "n_orders_3m"]].copy()
                item["architecture"], item["feature_set"], item["seed"] = architecture, definition, seed
                item["churn_probability"] = score
                individuals.append(item)
                fitted += 1
            model_path = model_directory / f"{architecture}_{definition}.joblib"
            joblib.dump({"models": models, "features": columns, "protocol": protocol}, model_path)
            model_paths.append(model_path)
            loaded = joblib.load(model_path)
            mean = np.mean(scores, axis=0)
            np.testing.assert_allclose(mean, np.mean([
                model.predict_proba(features[definition][columns])[:, 1] for model in loaded["models"]
            ], axis=0), atol=1e-14, rtol=0)
            item = features[definition][["StoreID", "snapshot", "n_orders_3m"]].copy()
            item["architecture"], item["feature_set"] = architecture, definition
            item["churn_probability"] = mean
            item["threshold"] = rules[architecture][definition]["threshold"]
            item["alert"] = item.churn_probability.ge(item.threshold).astype(int)
            predictions.append(item)
            print(f"Refitted {architecture} {definition}; {fitted}/20 fits", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    baseline_checks = []
    for architecture, prior in [("random_forest", RF_ORIGINAL), ("hist_gradient_boosting", ORIGINAL)]:
        saved_checkpoint = json.loads((prior / "prediction_checkpoint.json").read_text())
        expected_hash = (saved_checkpoint["predictions_sha256"] if architecture == "random_forest"
                         else saved_checkpoint["output_sha256"]["unlabeled_predictions.csv"])
        checked_hash(prior / "unlabeled_predictions.csv", expected_hash)
        saved = pd.read_csv(prior / "unlabeled_predictions.csv", float_precision="round_trip").sort_values("StoreID")
        current = predictions[predictions.architecture.eq(architecture) & predictions.feature_set.eq("subtotal")].sort_values("StoreID")
        np.testing.assert_array_equal(current.StoreID, saved.StoreID)
        np.testing.assert_allclose(current.churn_probability, saved.churn_probability, atol=1e-10, rtol=0)
        baseline_checks.append({"architecture": architecture,
                               "max_absolute_score_error": float(abs(current.churn_probability.to_numpy() - saved.churn_probability.to_numpy()).max())})
    predictions.to_csv(output / "unlabeled_predictions.csv", index=False)
    pd.concat(individuals, ignore_index=True).to_csv(output / "individual_seed_predictions.csv", index=False)
    write_json(output / "prediction_checkpoint.json", {
        "saved_at_utc": now(), "test_outcomes_previously_inspected": True, "models_fitted": fitted,
        "baseline_checks": baseline_checks,
        "protocol_sha256": file_sha256(output / "frozen_protocol.json"),
        "predictions_sha256": file_sha256(output / "unlabeled_predictions.csv"),
        "model_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in model_paths},
    })
    # Existing outcome reuse begins only here. This does not create an untouched test.
    label_path = ORIGINAL / "final_test_predictions.csv"
    original_manifest = json.loads((ORIGINAL / "evaluation_manifest.json").read_text())
    checked_hash(label_path, original_manifest["output_sha256"][label_path.name])
    labels = pd.read_csv(label_path)[["StoreID", "snapshot", "actual_churn"]]
    cases = attach_labels(predictions, labels)
    cases["fold"] = "october_2013"
    metrics = []
    for architecture, group in cases.groupby("architecture"):
        table = metrics_at_cutoffs(group, rules[architecture])
        table = table[table.fold.eq("october_2013")].copy()
        table["architecture"] = architecture
        metrics.append(table)
    metrics = pd.concat(metrics, ignore_index=True)
    cases.to_csv(output / "final_test_predictions.csv", index=False)
    metrics.to_csv(output / "final_test_metrics.csv", index=False)
    files = list(output.iterdir())
    write_json(output / "evaluation_manifest.json", {
        "completed_at_utc": now(), "models_fitted": fitted, "test_outcomes_previously_inspected": True,
        "independent_untouched_test": False, "thresholds_changed_using_test_outcomes": False,
        "test_rows": 492, "test_churners": 46,
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [
            preparation / "preparation_manifest.json", preparation / "development_subtotal.csv",
            preparation / "development_totaldue.csv", comparison / "experiment_manifest.json",
            comparison / "threshold_tradeoffs.csv", feature_path, clean_path, label_path]},
        "code_sha256": {str(Path(__file__).relative_to(ROOT)): file_sha256(Path(__file__))},
        "output_sha256": {p.name: file_sha256(p) for p in files},
    })
    print(metrics[metrics.subgroup.eq("overall")][["architecture", "feature_set", "threshold",
          "recall", "precision", "tp", "fn", "fp"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    run(arguments.preparation.resolve(), arguments.comparison.resolve(), arguments.run_id)

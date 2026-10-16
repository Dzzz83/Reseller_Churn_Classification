"""Freeze and score the October holdout before a separate label-reveal stage."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_companyx_features import clean_extracts, period_totals
from companyx_readonly import query
from compare_seven_ten_features import FEATURE_SETS

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.model_settings import BASELINE_GRADIENT_BOOSTING, MODEL_SEEDS
from reseller_churn.config.validation_settings import (
    EXPECTED_FINAL_TEST_ROWS,
    FINAL_TEST_SNAPSHOT,
)
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.evaluation.metrics import (
    calculate_ranking_metrics,
    calculate_threshold_metrics,
)
from reseller_churn.evaluation.selection import select_for_precision_floor
from reseller_churn.features.feature_pipeline import FeaturePipeline
from reseller_churn.modeling.development import validate_labeled_development_data
from reseller_churn.modeling.model_factory import ModelFactory

OUTPUT = ROOT / "results/final_evaluation/gb10_grouped_20261011"
MODEL_PATH = ROOT / "models/provisional/gb10_grouped_20261011/ensemble.joblib"
SOURCE = ROOT / "results/debugging/seven_vs_ten_features"
TRAINING_PATH = ROOT / "results/debugging/companyx_features/development_features.csv"
FEATURES = FEATURE_SETS["ten"]
FLOORS = (.20, .30, .40, .50)
PRIMARY_FLOOR = .30


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def frozen_rules(tradeoffs):
    """Derive only the predeclared rules from saved development predictions."""
    candidates = tradeoffs[
        tradeoffs.validation_method.eq("grouped_kfold")
        & tradeoffs.architecture.eq("hist_gradient_boosting")
        & tradeoffs.feature_set.eq("ten") & tradeoffs.score_kind.eq("ensemble")
    ].dropna(axis=1, how="all")
    rules = []
    for floor in FLOORS:
        selected = select_for_precision_floor(candidates, floor)
        if selected["status"] != "selected":
            raise ValueError(f"Predeclared precision floor {floor} is infeasible in development.")
        rules.append({
            "minimum_development_precision": floor, "threshold": selected["threshold"],
            "role": "primary" if floor == PRIMARY_FLOOR else "secondary",
            "development_mean_recall": selected["mean_recall"],
            "development_mean_precision": selected["mean_precision"],
        })
    return rules


def discount_features(keys, headers, lines, snapshot):
    """Only the three frozen discounts, from orders strictly before prediction."""
    snapshot = pd.Timestamp(snapshot)
    if keys.StoreID.duplicated().any():
        raise ValueError("Duplicate final-test reseller keys.")
    if not headers.OrderDate.lt(snapshot).all():
        raise ValueError("Future order supplied to final feature construction.")
    details = lines.merge(
        headers[["SalesOrderID", "StoreID", "OrderDate"]], on="SalesOrderID",
        validate="many_to_one", how="left",
    )
    if details.StoreID.isna().any():
        raise ValueError("Orphan final history line.")
    history_start = snapshot - pd.DateOffset(months=12)
    recent_start = snapshot - pd.DateOffset(months=3)
    previous_start = snapshot - pd.DateOffset(months=6)
    rates = []
    for start, end in [(history_start, snapshot), (recent_start, snapshot), (previous_start, recent_start)]:
        selected_headers = headers[headers.OrderDate.ge(start) & headers.OrderDate.lt(end)]
        selected_lines = details[details.OrderDate.ge(start) & details.OrderDate.lt(end)]
        rates.append(period_totals(selected_headers, selected_lines))
    annual, recent, previous = rates
    result = keys[["StoreID"]].copy()
    result["weighted_discount_3m"] = result.StoreID.map(recent["weighted_discount"])
    result["discount_change_3m"] = (
        result.weighted_discount_3m - result.StoreID.map(previous["weighted_discount"])
    )
    result["discounted_line_share_12m"] = result.StoreID.map(annual["discounted_share"])
    return result


def historical_extract():
    """Retrieve only pre-prediction orders; do not query the test outcome window."""
    headers = query("""
SELECT c.StoreID,h.SalesOrderID,CONVERT(varchar(23),h.OrderDate,121) AS OrderDate,
       h.SubTotal,h.SalesPersonID,h.TerritoryID
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20131001'
ORDER BY h.SalesOrderID;
""")
    headers["OrderDate"] = pd.to_datetime(headers["OrderDate"])
    lines = query("""
SELECT d.SalesOrderID,d.SalesOrderDetailID,d.ProductID,d.OrderQty,
       d.UnitPrice,d.UnitPriceDiscount,d.LineTotal
FROM Sales.SalesOrderDetail d JOIN Sales.SalesOrderHeader h ON h.SalesOrderID=d.SalesOrderID
JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20131001'
ORDER BY d.SalesOrderID,d.SalesOrderDetailID;
""")
    headers.to_csv(OUTPUT / "history_headers_raw.csv", index=False)
    lines.to_csv(OUTPUT / "history_lines_raw.csv", index=False)
    return headers, lines


def prepare():
    """Write an immutable methodology, refit, and checkpoint unlabeled predictions."""
    if OUTPUT.exists():
        raise ValueError("Final-evaluation directory already exists; preserve the frozen run.")
    tradeoffs = pd.read_csv(SOURCE / "threshold_tradeoffs.csv")
    prior = json.loads((SOURCE / "experiment_manifest.json").read_text())
    for path, expected in prior["code_sha256"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f"Development code changed: {path}")
    if file_sha256(TRAINING_PATH) != prior["input_sha256"][str(TRAINING_PATH.relative_to(ROOT))]:
        raise ValueError("Training feature data changed.")
    if file_sha256(SOURCE / "threshold_tradeoffs.csv") != prior["output_sha256"]["threshold_tradeoffs.csv"]:
        raise ValueError("Saved development threshold table changed.")
    data = validate_labeled_development_data(pd.read_csv(TRAINING_PATH))
    window = PredictionWindow.from_date(FINAL_TEST_SNAPSHOT)
    for snapshot in data.snapshot.unique():
        if PredictionWindow.from_date(snapshot).label_end > window.snapshot:
            raise ValueError("Training label unavailable at final prediction date.")
    code_paths = [
        Path(__file__), ROOT / "debug/build_companyx_features.py", ROOT / "debug/companyx_readonly.py",
        ROOT / "debug/compare_seven_ten_features.py", ROOT / "reseller_churn/modeling/model_factory.py",
        ROOT / "reseller_churn/data/churn_labels.py", ROOT / "reseller_churn/data/eligibility.py",
        ROOT / "reseller_churn/data/prediction_window.py", ROOT / "reseller_churn/features/feature_pipeline.py",
        ROOT / "reseller_churn/evaluation/metrics.py", ROOT / "reseller_churn/evaluation/selection.py",
    ]
    input_paths = [
        TRAINING_PATH, SOURCE / "experiment_manifest.json", SOURCE / "threshold_tradeoffs.csv",
        ROOT / "datasets/orders_clean.csv", ROOT / "datasets/stores_clean.csv",
        ROOT / "datasets/processed/ml_test_features.csv",
    ]
    protocol = {
        "frozen_at_utc": utc_now(), "architecture": "hist_gradient_boosting",
        "features": FEATURES, "parameters": asdict(BASELINE_GRADIENT_BOOSTING),
        "early_stopping": False, "preprocessing": "native missing-value handling",
        "imbalance_strategy": "balanced training sample weights", "seeds": list(MODEL_SEEDS),
        "seed_aggregation": "mean predicted probability", "threshold_source": "grouped_kfold",
        "rules": frozen_rules(tradeoffs), "training_rows": len(data),
        "training_snapshots": sorted(data.snapshot.unique().tolist()),
        "training_churn_cases": int(data.churn.sum()), "test_snapshot": FINAL_TEST_SNAPSHOT,
        "test_label_end_exclusive": str(window.label_end.date()), "expected_test_rows": EXPECTED_FINAL_TEST_ROWS,
        "authorization": {
            "final_evaluation_requested": "User: run the final testing evaluation.",
            "frozen_rule_choice": "User selected grouped 30% primary plus fixed 20%, 40%, 50% secondary rules.",
        },
        "test_outcomes_accessed_at_freeze": False,
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in input_paths},
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "Grouped development thresholds do not guarantee minimum test precision",
            "Order-time availability assumed; recording-time versions unverified",
            "Existing purchase/cleaning contract including accepted current cancelled orders retained",
            "No threshold or architecture will be selected using final labels",
        ],
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "frozen_protocol.json", protocol)
    headers, lines = historical_extract()
    orders = pd.read_csv(ROOT / "datasets/orders_clean.csv", parse_dates=["OrderDate"])
    history = orders[orders.OrderDate.lt(window.snapshot)].copy()
    headers, lines, excluded = clean_extracts(headers, lines, history)
    stores = pd.read_csv(ROOT / "datasets/stores_clean.csv")
    base = FeaturePipeline().build(history, stores, [FINAL_TEST_SNAPSHOT])
    reference = pd.read_csv(ROOT / "datasets/processed/ml_test_features.csv", parse_dates=["snapshot"])
    base = base.sort_values("StoreID").reset_index(drop=True)
    reference = reference.sort_values("StoreID").reset_index(drop=True)
    if len(base) != EXPECTED_FINAL_TEST_ROWS or "churn" in reference.columns:
        raise ValueError("Unexpected final-test population or labeled feature reference.")
    pd.testing.assert_frame_equal(base, reference, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-5)
    discounts = discount_features(base[["StoreID"]], headers, lines, window.snapshot)
    final_features = base.merge(discounts, on="StoreID", validate="one_to_one")
    final_features.to_csv(OUTPUT / "final_test_features.csv", index=False)
    weights = compute_sample_weight("balanced", data.churn)
    models, scores = [], []
    for seed in MODEL_SEEDS:
        model = ModelFactory.create_gradient_boosting(BASELINE_GRADIENT_BOOSTING, seed)
        model.fit(data[FEATURES], data.churn, sample_weight=weights)
        models.append(model)
        scores.append(model.predict_proba(final_features[FEATURES])[:, 1])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "protocol": protocol}, MODEL_PATH)
    saved = joblib.load(MODEL_PATH)
    mean_score = np.mean(scores, axis=0)
    reloaded_score = np.mean([
        model.predict_proba(final_features[FEATURES])[:, 1] for model in saved["models"]
    ], axis=0)
    np.testing.assert_allclose(mean_score, reloaded_score, rtol=0, atol=1e-14)
    predictions = final_features[["StoreID", "snapshot", "n_orders_3m"]].copy()
    predictions["churn_probability"] = mean_score
    for rule in protocol["rules"]:
        key = f"alert_precision_{int(rule['minimum_development_precision'] * 100):02d}"
        predictions[key] = predictions.churn_probability.ge(rule["threshold"]).astype(int)
    predictions.to_csv(OUTPUT / "unlabeled_predictions.csv", index=False)
    individual = pd.DataFrame({"StoreID": final_features.StoreID})
    for seed, probabilities in zip(MODEL_SEEDS, scores):
        individual[f"score_seed_{seed}"] = probabilities
    individual.to_csv(OUTPUT / "individual_seed_scores.csv", index=False)
    write_json(OUTPUT / "prediction_checkpoint.json", {
        "predictions_saved_at_utc": utc_now(), "test_outcomes_accessed": False,
        "models_fitted": len(models), "test_rows": len(predictions),
        "clean_history_orders": len(headers), "clean_history_lines": len(lines),
        "excluded_history_order_ids": excluded.SalesOrderID.astype(int).tolist(),
        "native_missing_counts": final_features[FEATURES].isna().sum().to_dict(),
        "model_path": str(MODEL_PATH.relative_to(ROOT)), "model_sha256": file_sha256(MODEL_PATH),
        "protocol_sha256": file_sha256(OUTPUT / "frozen_protocol.json"),
        "output_sha256": {
            name: file_sha256(OUTPUT / name) for name in [
                "history_headers_raw.csv", "history_lines_raw.csv", "final_test_features.csv",
                "unlabeled_predictions.csv", "individual_seed_scores.csv",
            ]
        },
    })
    print(f"Frozen GB10 protocol; fitted {len(models)} models and saved {len(predictions)} unlabeled final predictions.")


def verify_checkpoint(protocol, checkpoint):
    if protocol["test_outcomes_accessed_at_freeze"] or checkpoint["test_outcomes_accessed"]:
        raise ValueError("Pre-evaluation checkpoint already included test outcomes.")
    if file_sha256(OUTPUT / "frozen_protocol.json") != checkpoint["protocol_sha256"]:
        raise ValueError("Frozen final protocol changed.")
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in protocol[category].items():
            if file_sha256(ROOT / name) != expected:
                raise ValueError(f"Frozen {category} changed: {name}")
    for name, expected in checkpoint["output_sha256"].items():
        if file_sha256(OUTPUT / name) != expected:
            raise ValueError(f"Prediction checkpoint changed: {name}")
    if file_sha256(ROOT / checkpoint["model_path"]) != checkpoint["model_sha256"]:
        raise ValueError("Frozen model bundle changed.")


def evaluate():
    """After hash guards, reveal labels once; never fit or choose new thresholds."""
    if (OUTPUT / "evaluation_manifest.json").exists():
        raise ValueError("Final evaluation is already complete; use saved reports.")
    protocol = json.loads((OUTPUT / "frozen_protocol.json").read_text())
    checkpoint = json.loads((OUTPUT / "prediction_checkpoint.json").read_text())
    verify_checkpoint(protocol, checkpoint)
    predictions = pd.read_csv(OUTPUT / "unlabeled_predictions.csv")
    if "churn" in predictions or "actual_churn" in predictions:
        raise ValueError("Prediction checkpoint already contains outcomes.")
    write_json(OUTPUT / "outcome_access_started.json", {
        "started_at_utc": utc_now(), "test_snapshot": protocol["test_snapshot"],
        "notice": "Protected outcomes are now accessed under the authorized frozen protocol; this test is no longer untouched.",
    })
    # Both source reconciliation and label construction occur only after the checkpoint.
    future = query("""
SELECT c.StoreID,h.SalesOrderID,CONVERT(varchar(23),h.OrderDate,121) AS OrderDate,
       h.SubTotal,SUM(CAST(d.OrderQty AS bigint)) AS qty,COUNT(*) AS n_lines
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
JOIN Sales.SalesOrderDetail d ON d.SalesOrderID=h.SalesOrderID
WHERE c.StoreID IS NOT NULL AND h.OrderDate >= '20131001' AND h.OrderDate < '20140401'
GROUP BY c.StoreID,h.SalesOrderID,h.OrderDate,h.SubTotal
ORDER BY h.SalesOrderID;
""")
    future.OrderDate = pd.to_datetime(future.OrderDate)
    future.to_csv(OUTPUT / "label_window_orders_source.csv", index=False)
    orders = pd.read_csv(ROOT / "datasets/orders_clean.csv", parse_dates=["OrderDate"])
    window = PredictionWindow.from_date(protocol["test_snapshot"])
    reference = window.select_label_period(orders)
    source = future.merge(reference, on="SalesOrderID", how="outer", validate="one_to_one", suffixes=("", "_reference"), indicator=True)
    if not source["_merge"].eq("both").all():
        raise ValueError("Final label-window order populations differ from the source.")
    for name in ["StoreID", "OrderDate", "qty", "n_lines"]:
        np.testing.assert_array_equal(source[name], source[f"{name}_reference"])
    np.testing.assert_allclose(source.SubTotal, source.SubTotal_reference, atol=1e-5, rtol=1e-10)
    labels = build_churn_labels(orders, window)
    cases = predictions.merge(labels, on="StoreID", how="outer", validate="one_to_one", indicator=True)
    if not cases["_merge"].eq("both").all() or len(cases) != EXPECTED_FINAL_TEST_ROWS:
        raise ValueError("Final label population differs from frozen predictions.")
    cases = cases.drop(columns="_merge").rename(columns={"churn": "actual_churn"})
    rows = []
    for rule in protocol["rules"]:
        for subgroup, group in [
            ("overall", cases), ("purchased_in_3m", cases[cases.n_orders_3m.gt(0)]),
            ("no_purchase_in_3m", cases[cases.n_orders_3m.eq(0)]),
        ]:
            if group.empty:
                continue
            target, probabilities = group.actual_churn, group.churn_probability
            metrics = calculate_threshold_metrics(target, probabilities, rule["threshold"])
            rows.append({
                **rule, "subgroup": subgroup, "rows": len(group), "churners": int(target.sum()),
                **metrics, **calculate_ranking_metrics(target, probabilities),
                "test_precision_meets_development_floor": metrics["precision"] >= rule["minimum_development_precision"],
            })
    metrics = pd.DataFrame(rows)
    cases.to_csv(OUTPUT / "final_test_predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "final_test_metrics.csv", index=False)
    primary_rule = next(rule for rule in protocol["rules"] if rule["role"] == "primary")
    alert = cases.churn_probability.ge(primary_rule["threshold"])
    cases["primary_outcome"] = np.select(
        [cases.actual_churn.eq(1) & alert, cases.actual_churn.eq(1) & ~alert,
         cases.actual_churn.eq(0) & alert],
        ["caught_churn", "missed_churn", "false_alert"], default="correct_non_churn",
    )
    cases.to_csv(OUTPUT / "primary_error_analysis.csv", index=False)
    write_json(OUTPUT / "evaluation_manifest.json", {
        "completed_at_utc": utc_now(), "protected_final_test_labels_accessed": True,
        "test_rows": len(cases), "test_churners": int(cases.actual_churn.sum()),
        "label_window_source_orders": len(future), "source_reconciled_with_clean_csv": True,
        "methodology_frozen_before_outcome_access": True, "thresholds_changed_after_outcome_access": False,
        "models_fitted_in_evaluation_stage": 0, "primary_precision_floor": PRIMARY_FLOOR,
        "protocol_sha256": file_sha256(OUTPUT / "frozen_protocol.json"),
        "checkpoint_sha256": file_sha256(OUTPUT / "prediction_checkpoint.json"),
        "outcome_access_marker_sha256": file_sha256(OUTPUT / "outcome_access_started.json"),
        "output_sha256": {name: file_sha256(OUTPUT / name) for name in [
            "label_window_orders_source.csv", "final_test_predictions.csv",
            "final_test_metrics.csv", "primary_error_analysis.csv",
        ]},
    })
    print(metrics[metrics.subgroup.eq("overall")][[
        "role", "minimum_development_precision", "threshold", "rows", "churners",
        "recall", "precision", "f1", "average_precision", "roc_auc", "tp", "fp", "fn",
    ]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "evaluate"], required=True)
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    else:
        evaluate()

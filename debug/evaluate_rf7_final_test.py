"""Evaluate the grouped-development RF7 rules on the already-inspected October test."""

import argparse
from dataclasses import asdict
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

from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.model_settings import BASELINE_RANDOM_FOREST, MODEL_SEEDS
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.evaluation.metrics import calculate_ranking_metrics, calculate_threshold_metrics
from reseller_churn.evaluation.selection import evaluate_threshold_tradeoffs, select_for_precision_floor
from reseller_churn.modeling.development import file_sha256, validate_labeled_development_data, write_json
from reseller_churn.modeling.model_factory import ModelFactory


SOURCE = ROOT / "results/debugging/seven_vs_ten_features"
ORIGINAL_TEST = ROOT / "results/final_evaluation/gb10_grouped_20261011"
TRAINING_PATH = ROOT / "results/debugging/companyx_features/development_features.csv"
FEATURES = list(FEATURE_SETS["pruned"])
FLOORS = (0.20, 0.30, 0.40, 0.50)


def check_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Saved source changed: {path.relative_to(ROOT)}")


def development_rules():
    """Check the saved grid against OOF scores, then select development-only cutoffs."""
    predictions = pd.read_csv(SOURCE / "validation_predictions.csv", float_precision="round_trip")
    cases = predictions[
        predictions.validation_method.eq("grouped_kfold")
        & predictions.architecture.eq("random_forest")
        & predictions.feature_set.eq("seven")
    ]
    if len(cases) != 1375 or cases.actual_churn.sum() != 212 or cases.fold.nunique() != 5:
        raise ValueError("Unexpected grouped development population.")
    if cases.groupby("StoreID").fold.nunique().gt(1).any():
        raise ValueError("A reseller appears in multiple validation folds.")
    folds = {
        fold: {"target": group.actual_churn, "probabilities": group.churn_probability}
        for fold, group in cases.groupby("fold")
    }
    grid = evaluate_threshold_tradeoffs("random_forest__seven", folds)
    saved = pd.read_csv(SOURCE / "threshold_tradeoffs.csv", float_precision="round_trip")
    saved = saved[
        saved.validation_method.eq("grouped_kfold") & saved.architecture.eq("random_forest")
        & saved.feature_set.eq("seven") & saved.score_kind.eq("ensemble")
    ].sort_values("threshold")
    pd.testing.assert_frame_equal(
        grid.reset_index(drop=True), saved[grid.columns].reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )
    rules = []
    for floor in FLOORS:
        selected = select_for_precision_floor(grid, floor)
        if selected["status"] != "selected":
            raise ValueError(f"Infeasible development precision floor: {floor}")
        rules.append({
            "minimum_development_precision": floor, "threshold": selected["threshold"],
            "role": "primary" if floor == 0.30 else "secondary",
            "development_mean_recall": selected["mean_recall"],
            "development_mean_precision": selected["mean_precision"],
        })
    return rules


def run(run_id):
    output = ROOT / "results/final_evaluation" / run_id
    model_path = ROOT / "models/provisional" / run_id / "ensemble.joblib"
    if output.exists() or model_path.parent.exists():
        raise ValueError("Run already exists; choose a new run ID.")
    manifest = json.loads((SOURCE / "experiment_manifest.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        check_hash(ROOT / name, expected)
    check_hash(TRAINING_PATH, manifest["input_sha256"][str(TRAINING_PATH.relative_to(ROOT))])
    for name in ["validation_predictions.csv", "threshold_tradeoffs.csv"]:
        check_hash(SOURCE / name, manifest["output_sha256"][name])
    original_checkpoint = json.loads((ORIGINAL_TEST / "prediction_checkpoint.json").read_text())
    feature_path = ORIGINAL_TEST / "final_test_features.csv"
    check_hash(feature_path, original_checkpoint["output_sha256"][feature_path.name])
    data = validate_labeled_development_data(pd.read_csv(TRAINING_PATH))
    features = pd.read_csv(feature_path, parse_dates=["snapshot"])
    reference = pd.read_csv(ROOT / "datasets/processed/ml_test_features.csv", parse_dates=["snapshot"])
    pd.testing.assert_frame_equal(
        features[["StoreID", "snapshot", *FEATURES]].sort_values("StoreID").reset_index(drop=True),
        reference[["StoreID", "snapshot", *FEATURES]].sort_values("StoreID").reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-5,
    )
    if len(data) != 1375 or data.churn.sum() != 212 or len(features) != 492:
        raise ValueError("Unexpected training or test population.")
    if not features.snapshot.eq(pd.Timestamp("2013-10-01")).all():
        raise ValueError("Unexpected final prediction snapshot.")
    for snapshot in data.snapshot.unique():
        if PredictionWindow.from_date(snapshot).label_end > pd.Timestamp("2013-10-01"):
            raise ValueError("Training label unavailable at prediction time.")
    protocol = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "random_forest", "features": FEATURES,
        "parameters": asdict(BASELINE_RANDOM_FOREST), "seeds": list(MODEL_SEEDS),
        "preprocessing": "training-fitted median imputation with missing indicators",
        "imbalance_strategy": "balanced training sample weights",
        "seed_aggregation": "mean predicted probability", "threshold_source": "grouped_kfold",
        "rules": development_rules(), "training_rows": len(data), "training_churn_cases": int(data.churn.sum()),
        "test_snapshot": "2013-10-01", "test_label_end_exclusive": "2014-04-01",
        "authorization": "User requested RF7 grouped-development rules evaluated on the final testing set, 2026-10-14.",
        "test_outcomes_previously_inspected": True,
        "limitations": [
            "October outcomes were inspected in the original GB10 evaluation and in this chat; this is a retrospective additional-model evaluation.",
            "No model settings or cutoffs are selected using October labels.",
            "Grouped development precision floors do not guarantee test precision.",
        ],
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in [
            TRAINING_PATH, SOURCE / "experiment_manifest.json", SOURCE / "threshold_tradeoffs.csv",
            SOURCE / "validation_predictions.csv", feature_path,
        ]},
        "code_sha256": {str(Path(__file__).relative_to(ROOT)): file_sha256(Path(__file__))},
    }
    output.mkdir(parents=True)
    write_json(output / "frozen_protocol.json", protocol)
    models, scores = [], []
    weights = compute_sample_weight("balanced", data.churn)
    for seed in MODEL_SEEDS:
        model = ModelFactory.create_random_forest(BASELINE_RANDOM_FOREST, seed)
        model.fit(data[FEATURES], data.churn, model__sample_weight=weights)
        models.append(model)
        scores.append(model.predict_proba(features[FEATURES])[:, 1])
    model_path.parent.mkdir(parents=True)
    joblib.dump({"models": models, "protocol": protocol}, model_path)
    saved = joblib.load(model_path)
    mean_scores = np.mean(scores, axis=0)
    np.testing.assert_allclose(mean_scores, np.mean([
        model.predict_proba(features[FEATURES])[:, 1] for model in saved["models"]
    ], axis=0), rtol=0, atol=1e-14)
    predictions = features[["StoreID", "snapshot", "n_orders_3m"]].copy()
    predictions["churn_probability"] = mean_scores
    for rule in protocol["rules"]:
        key = f"alert_precision_{int(rule['minimum_development_precision'] * 100):02d}"
        predictions[key] = predictions.churn_probability.ge(rule["threshold"]).astype(int)
    predictions.to_csv(output / "unlabeled_predictions.csv", index=False)
    individual = pd.DataFrame({"StoreID": features.StoreID})
    for seed, score in zip(MODEL_SEEDS, scores):
        individual[f"score_seed_{seed}"] = score
    individual.to_csv(output / "individual_seed_scores.csv", index=False)
    write_json(output / "prediction_checkpoint.json", {
        "predictions_saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_outcomes_previously_inspected": True, "models_fitted": len(models),
        "model_sha256": file_sha256(model_path),
        "protocol_sha256": file_sha256(output / "frozen_protocol.json"),
        "predictions_sha256": file_sha256(output / "unlabeled_predictions.csv"),
    })
    # Access the existing October outcomes only after recording RF settings and scores.
    original_evaluation = json.loads((ORIGINAL_TEST / "evaluation_manifest.json").read_text())
    label_path = ORIGINAL_TEST / "final_test_predictions.csv"
    check_hash(label_path, original_evaluation["output_sha256"][label_path.name])
    labels = pd.read_csv(label_path, parse_dates=["snapshot"])[["StoreID", "snapshot", "actual_churn"]]
    cases = predictions.merge(labels, on=["StoreID", "snapshot"], how="outer", validate="one_to_one", indicator=True)
    if not cases._merge.eq("both").all() or len(cases) != 492 or cases.actual_churn.sum() != 46:
        raise ValueError("Test label population differs from saved original evaluation.")
    cases = cases.drop(columns="_merge")
    rows = []
    for rule in protocol["rules"]:
        for subgroup, group in [
            ("overall", cases), ("purchased_in_3m", cases[cases.n_orders_3m.gt(0)]),
            ("no_purchase_in_3m", cases[cases.n_orders_3m.eq(0)]),
        ]:
            metrics = calculate_threshold_metrics(group.actual_churn, group.churn_probability, rule["threshold"])
            if metrics["tp"] + metrics["fn"] != int(group.actual_churn.sum()):
                raise ValueError("Confusion counts do not reconcile.")
            rows.append({**rule, "subgroup": subgroup, "rows": len(group),
                         "churners": int(group.actual_churn.sum()), **metrics,
                         **calculate_ranking_metrics(group.actual_churn, group.churn_probability)})
    metrics = pd.DataFrame(rows)
    cases.to_csv(output / "final_test_predictions.csv", index=False)
    metrics.to_csv(output / "final_test_metrics.csv", index=False)
    write_json(output / "evaluation_manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_outcomes_previously_inspected": True, "independent_untouched_test": False,
        "thresholds_changed_using_test_outcomes": False, "test_rows": len(cases), "test_churners": 46,
        "label_source": str(label_path.relative_to(ROOT)), "label_source_sha256": file_sha256(label_path),
        "output_sha256": {name: file_sha256(output / name) for name in ["final_test_predictions.csv", "final_test_metrics.csv"]},
    })
    print(metrics[metrics.subgroup.eq("overall")][[
        "minimum_development_precision", "threshold", "recall", "precision", "tp", "fn", "fp", "alert_count",
    ]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    if not arguments.run_id or Path(arguments.run_id).name != arguments.run_id or arguments.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    run(arguments.run_id)

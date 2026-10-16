"""Check retrospective revenue artifacts without training or querying outcomes."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compare_subtotal_totaldue import ROOT, checked_hash, metrics_at_cutoffs
from evaluate_revenue_final_test import attach_labels
from prepare_subtotal_totaldue import REVENUE_COLUMNS
from verify_subtotal_totaldue_outputs import assert_report_equal


def verify(output):
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    protocol = json.loads((output / "frozen_protocol.json").read_text())
    checkpoint = json.loads((output / "prediction_checkpoint.json").read_text())
    if manifest["models_fitted"] != 20 or manifest["independent_untouched_test"] or manifest["thresholds_changed_using_test_outcomes"]:
        raise ValueError("Unexpected evaluation scope")
    if not manifest["test_outcomes_previously_inspected"]:
        raise ValueError("Missing prior-inspection disclosure")
    for name, expected in manifest["output_sha256"].items():
        checked_hash(output / name, expected)
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in manifest[category].items():
            checked_hash(ROOT / name, expected)
    checked_hash(output / "frozen_protocol.json", checkpoint["protocol_sha256"])
    checked_hash(output / "unlabeled_predictions.csv", checkpoint["predictions_sha256"])
    if datetime.fromisoformat(protocol["recorded_at_utc"]) > datetime.fromisoformat(checkpoint["saved_at_utc"]):
        raise ValueError("Invalid protocol/checkpoint chronology")
    if datetime.fromisoformat(checkpoint["saved_at_utc"]) > datetime.fromisoformat(manifest["completed_at_utc"]):
        raise ValueError("Invalid evaluation chronology")

    def read(name):
        return pd.read_csv(output / name, float_precision="round_trip")

    features = {name: read(f"features_{name}.csv") for name in ["subtotal", "totaldue"]}
    unchanged = features["subtotal"].columns.difference(REVENUE_COLUMNS)
    pd.testing.assert_frame_equal(features["subtotal"][unchanged], features["totaldue"][unchanged], check_exact=True)
    predictions = read("unlabeled_predictions.csv")
    individual = read("individual_seed_predictions.csv")
    if len(predictions) != 1968 or len(individual) != 9840:
        raise ValueError("Unexpected prediction coverage")
    if "actual_churn" in predictions or "churn" in predictions:
        raise ValueError("Unlabeled checkpoint contains outcomes")
    for (architecture, definition), cases in predictions.groupby(["architecture", "feature_set"]):
        model_path = ROOT / "models/provisional" / output.name / f"{architecture}_{definition}.joblib"
        checked_hash(model_path, checkpoint["model_sha256"][str(model_path.relative_to(ROOT))])
        bundle = joblib.load(model_path)
        scores = np.mean([model.predict_proba(features[definition][bundle["features"]])[:, 1]
                          for model in bundle["models"]], axis=0)
        cases = cases.sort_values("StoreID")
        np.testing.assert_array_equal(cases.StoreID, features[definition].StoreID)
        np.testing.assert_allclose(cases.churn_probability, scores, atol=1e-14, rtol=0)
        seed_cases = individual[individual.architecture.eq(architecture) & individual.feature_set.eq(definition)]
        means = seed_cases.groupby(["StoreID", "snapshot"]).churn_probability.mean().reset_index().sort_values("StoreID")
        np.testing.assert_allclose(cases.churn_probability, means.churn_probability, atol=1e-14, rtol=0)
        threshold = protocol["rules"][architecture][definition]["threshold"]
        assert cases.threshold.eq(threshold).all()
        np.testing.assert_array_equal(cases.alert, cases.churn_probability.ge(threshold).astype(int))
    saved = read("final_test_predictions.csv")
    labels = saved[["StoreID", "snapshot", "actual_churn"]].drop_duplicates()
    expected_cases = attach_labels(predictions, labels)
    expected_cases["fold"] = "october_2013"
    assert_report_equal(saved, expected_cases, ["architecture", "feature_set", "StoreID", "snapshot"])
    reported = read("final_test_metrics.csv")
    for architecture, cases in saved.groupby("architecture"):
        expected = metrics_at_cutoffs(cases, protocol["rules"][architecture])
        expected = expected[expected.fold.eq("october_2013")]
        assert_report_equal(reported[reported.architecture.eq(architecture)], expected, ["feature_set", "subgroup"])
    print("Verified hashes, model reload scores, seed means, fixed decisions, matched cohorts,")
    print("and all overall/subgroup metrics. Prior outcome inspection remains disclosed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output.resolve())

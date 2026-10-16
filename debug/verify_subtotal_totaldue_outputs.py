"""Verify a saved grouped revenue experiment without training or database access."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_subtotal_totaldue import (
    ROOT, average_seeds, checked_hash, matched_cases, metrics_at_cutoffs, summarize,
)


def assert_report_equal(actual, expected, keys):
    """Compare report rows after ordering keys, allowing only CSV roundoff."""
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True)[expected.columns],
        expected.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-12, rtol=0,
    )


def verify(output):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    if manifest["models_fitted"] != 100 or manifest["final_test_outcomes_accessed"]:
        raise ValueError("Unexpected execution scope")
    for name, expected in manifest["output_sha256"].items():
        checked_hash(output / name, expected)
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in manifest[category].items():
            checked_hash(ROOT / name, expected)
    read = lambda name: pd.read_csv(output / name, float_precision="round_trip")
    membership = read("fold_membership.csv")
    validation = membership[membership.role.eq("validation")]
    if len(validation) != 1375 or validation[["StoreID", "snapshot"]].duplicated().any():
        raise ValueError("Invalid held-out coverage")
    for _, fold in membership.groupby("fold"):
        train = fold[fold.role.eq("training")]
        cases = fold[fold.role.eq("validation")]
        if set(train.StoreID) & set(cases.StoreID):
            raise ValueError("Reseller overlap")
    individual = read("individual_seed_predictions.csv")
    ensemble = read("validation_predictions.csv")
    if len(individual) != 27500 or len(ensemble) != 5500:
        raise ValueError("Unexpected prediction counts")
    assert_report_equal(ensemble, average_seeds(individual), ["architecture", "feature_set", "fold", "StoreID", "snapshot"])
    selections_table = read("selection_summary.csv")
    grids = read("threshold_tradeoffs.csv")
    metrics = read("fold_and_subgroup_metrics.csv")
    seed_metrics = read("seed_metrics_at_ensemble_cutoff.csv")
    diagnostic = read("metrics_at_subtotal_cutoff.csv")
    paired = read("paired_cases.csv")
    metric_keys = ["feature_set", "fold", "subgroup"]
    for architecture, cases in ensemble.groupby("architecture"):
        selections, summary, grid = summarize(cases)
        observed_summary = selections_table[selections_table.architecture.eq(architecture)]
        assert_report_equal(observed_summary, summary, ["feature_set"])
        assert_report_equal(grids[grids.architecture.eq(architecture)], grid, ["feature_set", "threshold"])
        assert_report_equal(metrics[metrics.architecture.eq(architecture)], metrics_at_cutoffs(cases, selections), metric_keys)
        for seed, group in individual[individual.architecture.eq(architecture)].groupby("seed"):
            observed = seed_metrics[seed_metrics.architecture.eq(architecture) & seed_metrics.seed.eq(seed)]
            assert_report_equal(observed, metrics_at_cutoffs(group, selections), metric_keys)
        if selections["subtotal"]["status"] == "selected":
            shared = {name: selections["subtotal"] for name in selections}
            assert_report_equal(diagnostic[diagnostic.architecture.eq(architecture)], metrics_at_cutoffs(cases, shared), metric_keys)
        expected_pair = matched_cases(cases, selections)
        if not expected_pair.empty:
            assert_report_equal(paired[paired.architecture.eq(architecture)], expected_pair, ["fold", "StoreID", "snapshot"])
        # Verify each configuration was evaluated on exactly the recorded held-out cases.
        for _, group in cases.groupby("feature_set"):
            expected = validation[["StoreID", "snapshot", "fold", "churn"]].rename(columns={"churn": "actual_churn"})
            assert_report_equal(group, expected, ["StoreID", "snapshot"])
    counts = paired.groupby(["architecture", "fold", "change_type"]).size().rename("cases").reset_index()
    assert_report_equal(read("prediction_change_counts.csv"), counts, ["architecture", "fold", "change_type"])
    if not np.isfinite(ensemble.churn_probability).all():
        raise ValueError("Non-finite scores")
    print("Verified input/code/output hashes, shared folds, seed averaging, selections,")
    print("fold/pooled/subgroup/seed metrics, fixed-cutoff diagnostics, and paired decisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output.resolve())

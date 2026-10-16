"""Verify nested saved splits, cutoff grids, decisions, and paired intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_subtotal_totaldue import ROOT, checked_hash, load_pair
from nested_revenue_comparison import (
    BOOTSTRAP_DRAWS, MODEL_SEEDS, REPEAT_SEEDS, bootstrap_differences,
    nested_splits, operating_rows, pooled_rows,
)
from reseller_churn.evaluation.selection import select_for_precision_floor
from verify_subtotal_totaldue_outputs import assert_report_equal


def verify_inner_grid(predictions, grid):
    """Reconstruct confusion counts directly for all cutoffs, independently."""
    thresholds = grid.threshold.to_numpy()
    recalls, precisions = [], []
    pooled = {name: np.zeros(len(grid), dtype=int) for name in ["tp", "fp", "fn", "tn"]}
    for fold, cases in predictions.groupby("inner_fold"):
        target = cases.actual_churn.to_numpy().astype(bool)
        alerted = cases.churn_probability.to_numpy()[None, :] >= thresholds[:, None]
        counts = {"tp": (alerted & target).sum(axis=1), "fp": (alerted & ~target).sum(axis=1),
                  "fn": (~alerted & target).sum(axis=1), "tn": (~alerted & ~target).sum(axis=1)}
        for name, values in counts.items():
            np.testing.assert_array_equal(grid[f"{fold}_{name}"], values)
            pooled[name] += values
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        recall = np.divide(tp, tp + fn, out=np.zeros(len(grid)), where=(tp + fn) != 0)
        precision = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) != 0)
        np.testing.assert_allclose(grid[f"{fold}_precision"], precision, atol=1e-12, rtol=0)
        np.testing.assert_allclose(grid[f"{fold}_recall"], recall, atol=1e-12, rtol=0)
        recalls.append(recall)
        precisions.append(precision)
    for name, values in pooled.items():
        np.testing.assert_array_equal(grid[f"pooled_{name}"], values)
    for name, values in [("mean_recall", np.mean(recalls, axis=0)), ("worst_fold_recall", np.min(recalls, axis=0)),
                         ("mean_precision", np.mean(precisions, axis=0))]:
        np.testing.assert_allclose(grid[name], values, atol=1e-12, rtol=0)


def verify(output, require_complete=False):
    protocol = json.loads((output / "protocol.json").read_text())
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in protocol[category].items():
            checked_hash(ROOT / name, expected)
    preparation_path = next(name for name in protocol["input_sha256"] if name.endswith("preparation_manifest.json"))
    data = load_pair((ROOT / preparation_path).parent)["subtotal"]
    all_outer, all_pooled, all_thresholds, completed = [], [], [], 0
    for seed in REPEAT_SEEDS:
        folder = output / f"repeat_{seed}"
        if not folder.exists():
            continue
        completion = json.loads((folder / "completion.json").read_text())
        assert completion["models_fitted"] == 600
        for name, expected in completion["output_sha256"].items():
            checked_hash(folder / name, expected)

        def read(name):
            return pd.read_csv(folder / name, float_precision="round_trip")

        members = read("memberships.csv")
        for outer_fold, (train, validation, inner) in enumerate(nested_splits(data, seed), start=1):
            for stage, inner_fold, left, right in [("outer", 0, train, validation), *[
                ("inner", number, a, b) for number, (a, b) in enumerate(inner, start=1)]]:
                for role, indices in [("training", left), ("validation", right)]:
                    actual = members[members.outer_fold.eq(outer_fold) & members.stage.eq(stage)
                                     & members.inner_fold.eq(inner_fold) & members.role.eq(role)]
                    assert_report_equal(actual, data.iloc[indices][["StoreID", "snapshot", "churn"]], ["StoreID", "snapshot"])
        individual, inner_predictions, outer = read("individual_predictions.csv"), read("inner_predictions.csv"), read("outer_predictions.csv")
        all_outer.append(outer)
        keys = ["stage", "repeat_seed", "outer_fold", "inner_fold", "architecture", "feature_set", "StoreID", "snapshot"]
        if individual[keys + ["seed"]].duplicated().any():
            raise ValueError("Duplicate seed score")
        grouped = individual.groupby(keys)
        if not grouped.seed.apply(lambda group: set(group) == set(MODEL_SEEDS)).all():
            raise ValueError("Incomplete seed score coverage")
        means = grouped.churn_probability.mean().reset_index()
        combined = pd.concat([inner_predictions, outer], ignore_index=True)
        assert_report_equal(combined, means, keys)
        grids, selections = read("inner_threshold_tradeoffs.csv"), read("selected_thresholds.csv")
        selections["repeat_seed"] = seed
        all_thresholds.append(selections)
        expected_metrics = []
        for (architecture, definition, outer_fold), cases in outer.groupby(["architecture", "feature_set", "outer_fold"]):
            mask = lambda frame: frame.architecture.eq(architecture) & frame.feature_set.eq(definition) & frame.outer_fold.eq(outer_fold)
            inner = inner_predictions[mask(inner_predictions)]
            grid = grids[mask(grids)].sort_values("threshold")
            verify_inner_grid(inner, grid)
            rule = select_for_precision_floor(grid, .30)
            saved_rule = selections[mask(selections)].iloc[0]
            assert rule["status"] == saved_rule.status
            assert cases.status.eq(rule["status"]).all()
            if rule["status"] == "selected":
                assert saved_rule.threshold == rule["threshold"]
                assert cases.threshold.eq(rule["threshold"]).all()
                np.testing.assert_array_equal(cases.alert, cases.churn_probability.ge(rule["threshold"]).astype(int))
            else:
                assert cases.threshold.isna().all() and cases.alert.isna().all()
            recorded_keys = data.iloc[nested_splits(data, seed)[outer_fold - 1][1]][["StoreID", "snapshot", "churn"]].rename(columns={"churn": "actual_churn"})
            assert_report_equal(cases, recorded_keys, ["StoreID", "snapshot"])
            expected_metrics.extend(operating_rows(cases, rule["threshold"], rule["status"], {
                "architecture": architecture, "feature_set": definition, "outer_fold": outer_fold, "repeat_seed": seed}))
        assert_report_equal(read("outer_metrics.csv"), pd.DataFrame(expected_metrics), ["architecture", "feature_set", "outer_fold", "subgroup"])
        pooled = read("pooled_metrics.csv")
        assert_report_equal(pooled, pooled_rows(outer), ["architecture", "feature_set", "subgroup"])
        pooled["repeat_seed"] = seed
        all_pooled.append(pooled)
        completed += 1
    if not completed or (require_complete and completed != 3):
        raise ValueError("Required completed repeats are absent")
    if (output / "analysis").exists():
        analysis = output / "analysis"
        manifest = json.loads((analysis / "manifest.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            checked_hash(analysis / name, expected)
        expected = bootstrap_differences(pd.concat(all_outer, ignore_index=True), BOOTSTRAP_DRAWS)
        actual = pd.read_csv(analysis / "paired_intervals.csv", float_precision="round_trip")
        assert_report_equal(actual, expected, ["architecture", "metric"])
        expected_pooled = pd.concat(all_pooled, ignore_index=True)
        expected_thresholds = pd.concat(all_thresholds, ignore_index=True)
        assert_report_equal(pd.read_csv(analysis / "repeat_pooled_metrics.csv", float_precision="round_trip"),
                            expected_pooled, ["repeat_seed", "architecture", "feature_set", "subgroup"])
        assert_report_equal(pd.read_csv(analysis / "all_selected_thresholds.csv", float_precision="round_trip"),
                            expected_thresholds, ["repeat_seed", "architecture", "feature_set", "outer_fold"])
        expected_stability = expected_thresholds.groupby(["architecture", "feature_set"]).threshold.agg(["min", "median", "max"]).reset_index()
        assert_report_equal(pd.read_csv(analysis / "threshold_stability.csv", float_precision="round_trip"),
                            expected_stability, ["architecture", "feature_set"])
        summary = pd.read_csv(analysis / "comparison_summary.csv", float_precision="round_trip")
        for architecture, group in expected_pooled[expected_pooled.subgroup.eq("overall")].groupby("architecture"):
            if not group.status.eq("selected").all():
                continue
            means = group.groupby("feature_set")[["recall", "precision"]].mean()
            row = summary[summary.architecture.eq(architecture)].iloc[0]
            for metric in ["recall", "precision"]:
                for definition in ["subtotal", "totaldue"]:
                    np.testing.assert_allclose(row[f"{definition}_{metric}"], means.loc[definition, metric], atol=1e-12, rtol=0)
                np.testing.assert_allclose(row[f"delta_{metric}"], means.loc["totaldue", metric] - means.loc["subtotal", metric], atol=1e-12, rtol=0)
    print(f"Verified {completed} repeats: hashes, nested memberships, seed means, all inner grids,")
    print("selected cutoffs, outer coverage/decisions/metrics, and available bootstrap intervals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    verify(args.output.resolve(), args.require_complete)

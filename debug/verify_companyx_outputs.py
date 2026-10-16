"""Reconcile saved experiment outputs without training or reading new labels."""

import ast
import json
import re

import numpy as np
import pandas as pd
from build_companyx_features import FEATURE_GROUPS, OUTPUT, ROOT

from reseller_churn.artifacts import file_sha256


def main():
    manifest = json.loads((OUTPUT / "experiment_manifest.json").read_text())
    for category in ["input_sha256", "code_sha256"]:
        for relative_path, expected in manifest[category].items():
            assert file_sha256(ROOT / relative_path) == expected, relative_path
    for name, expected in manifest["output_sha256"].items():
        assert file_sha256(OUTPUT / name) == expected, name
    cleaning = json.loads((OUTPUT / "cleaning_manifest.json").read_text())
    for name, expected in cleaning["input_sha256"].items():
        assert file_sha256(ROOT / name) == expected, name

    predictions = pd.read_csv(OUTPUT / "validation_predictions.csv")
    metrics = pd.read_csv(OUTPUT / "fold_and_subgroup_metrics.csv")
    assert len(predictions) == 4 * 683
    assert len(metrics) == 4 * 2 * 3
    for _, row in metrics.iterrows():
        cases = predictions[
            predictions.feature_set.eq(row.feature_set) & predictions.fold.eq(row.fold)
        ]
        if row.subgroup == "purchased_in_3m":
            cases = cases[cases.n_orders_3m.gt(0)]
        elif row.subgroup == "no_purchase_in_3m":
            cases = cases[cases.n_orders_3m.eq(0)]
        else:
            assert row.subgroup == "overall", row.subgroup
        positive = cases.actual_churn.eq(1)
        alert = cases.churn_probability.ge(row.threshold)
        actual = [
            int((positive & alert).sum()), int((~positive & alert).sum()),
            int((positive & ~alert).sum()), int((~positive & ~alert).sum()),
        ]
        assert actual == [row.tp, row.fp, row.fn, row.tn]
        assert len(cases) == row.rows
        np.testing.assert_allclose(row.recall, actual[0] / max(1, actual[0] + actual[2]))
        np.testing.assert_allclose(row.precision, actual[0] / max(1, actual[0] + actual[1]))
    overall = metrics[metrics.subgroup.eq("overall")]
    assert overall.precision.ge(.30).all()
    for name, selected in manifest["selections"].items():
        rows = overall[overall.feature_set.eq(name)]
        np.testing.assert_allclose(rows.recall.mean(), selected["mean_recall"])
        np.testing.assert_allclose(rows.precision.mean(), selected["mean_precision"])
        assert rows.tp.sum() == selected["pooled_tp"]
        assert rows.fp.sum() == selected["pooled_fp"]

    paired = pd.read_csv(OUTPUT / "paired_cases.csv")
    assert len(paired) == 3 * 683
    for name in FEATURE_GROUPS:
        for fold in overall.fold.unique():
            cases = paired[paired.feature_group.eq(name) & paired.fold.eq(fold)]
            baseline = overall[overall.feature_set.eq("baseline") & overall.fold.eq(fold)].iloc[0]
            candidate = overall[overall.feature_set.eq(name) & overall.fold.eq(fold)].iloc[0]
            counts = cases.change_type.value_counts()
            assert candidate.tp - baseline.tp == (
                counts.get("newly_caught_churn", 0) - counts.get("newly_missed_churn", 0)
            )
            assert candidate.fp - baseline.fp == (
                counts.get("added_false_alert", 0) - counts.get("removed_false_alert", 0)
            )
    assert manifest["protected_final_test_labels_accessed"] is False
    for path in (ROOT / "debug").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
        assert all(line.rstrip() == line for line in path.read_text().splitlines()), path
    report = ROOT / "docs/debugging.md"
    assert all(line.rstrip() == line for line in report.read_text().splitlines())
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        if "://" not in target:
            assert (report.parent / target.split("#")[0]).exists(), target
    print("Verified hashes, confusion counts, subgroup metrics, selections, paired changes, report links, syntax and whitespace.")


if __name__ == "__main__":
    main()

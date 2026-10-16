"""Independent accounting checks for the saved two-validation comparison."""

import ast
import json
import re

import numpy as np
import pandas as pd

from compare_seven_ten_features import OUTPUT, ROOT
from reseller_churn.config.validation_settings import LABELED_SNAPSHOTS
from reseller_churn.modeling.development import file_sha256


def main():
    manifest = json.loads((OUTPUT / "experiment_manifest.json").read_text())
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in manifest[category].items():
            assert file_sha256(ROOT / name) == expected, name
    for name, expected in manifest["output_sha256"].items():
        assert file_sha256(OUTPUT / name) == expected, name
    assert manifest["models_fitted"] == 154
    assert manifest["protected_final_test_labels_accessed"] is False
    membership = pd.read_csv(OUTPUT / "fold_membership.csv")
    assert set(membership.snapshot) == set(LABELED_SNAPSHOTS)
    grouped = membership[membership.validation_method.eq("grouped_kfold")]
    for _, fold in grouped.groupby("fold"):
        training = fold[fold.role.eq("training")]
        validation = fold[fold.role.eq("validation")]
        assert set(training.StoreID).isdisjoint(validation.StoreID)
    held_out = grouped[grouped.role.eq("validation")]
    assert len(held_out) == 1375
    assert not held_out[["StoreID", "snapshot"]].duplicated().any()
    assert held_out.groupby("StoreID").fold.nunique().eq(1).all()
    temporal = membership[membership.validation_method.eq("temporal")]
    for _, fold in temporal.groupby("fold"):
        latest_training = pd.to_datetime(fold.loc[fold.role.eq("training"), "snapshot"]).max()
        prediction_date = pd.to_datetime(fold.loc[fold.role.eq("validation"), "snapshot"]).min()
        assert latest_training + pd.DateOffset(months=6) <= prediction_date

    predictions = pd.read_csv(OUTPUT / "validation_predictions.csv")
    individual = pd.read_csv(OUTPUT / "individual_seed_predictions.csv")
    assert len(predictions) == 6 * (683 + 1375)
    keys = ["validation_method", "architecture", "feature_set", "fold", "StoreID", "snapshot", "actual_churn", "n_orders_3m"]
    assert not predictions[keys[:6]].duplicated().any()
    averaged = individual.groupby(keys, as_index=False).churn_probability.mean()
    reconciled = predictions.merge(averaged, on=keys, validate="one_to_one", suffixes=("", "_mean"))
    assert len(reconciled) == len(predictions)
    np.testing.assert_allclose(reconciled.churn_probability, reconciled.churn_probability_mean, atol=1e-14)

    metrics = pd.read_csv(OUTPUT / "fold_and_subgroup_metrics.csv", dtype={"seed": str})
    for _, row in metrics.iterrows():
        source = predictions if row.score_kind == "ensemble" else individual[individual.seed.eq(int(row.seed))]
        cases = source[
            source.validation_method.eq(row.validation_method)
            & source.architecture.eq(row.architecture)
            & source.feature_set.eq(row.feature_set) & source.fold.eq(row.fold)
        ]
        if row.subgroup == "purchased_in_3m":
            cases = cases[cases.n_orders_3m.gt(0)]
        elif row.subgroup == "no_purchase_in_3m":
            cases = cases[cases.n_orders_3m.eq(0)]
        else:
            assert row.subgroup == "overall"
        positive, alert = cases.actual_churn.eq(1), cases.churn_probability.ge(row.threshold)
        counts = [int((positive & alert).sum()), int((~positive & alert).sum()),
                  int((positive & ~alert).sum()), int((~positive & ~alert).sum())]
        assert counts == [row.tp, row.fp, row.fn, row.tn]
        assert sum(counts) == row.rows
        np.testing.assert_allclose(row.recall, counts[0] / max(1, counts[0] + counts[2]))
        np.testing.assert_allclose(row.precision, counts[0] / max(1, counts[0] + counts[1]))
    summary = pd.read_csv(OUTPUT / "selection_summary.csv", dtype={"seed": str})
    assert len(summary) == 56
    overall = metrics[metrics.subgroup.eq("overall")]
    for _, row in summary.iterrows():
        cases = overall[
            overall.validation_method.eq(row.validation_method) & overall.architecture.eq(row.architecture)
            & overall.feature_set.eq(row.feature_set) & overall.score_kind.eq(row.score_kind)
            & overall.seed.eq(row.seed)
        ]
        if row.status != "selected":
            assert cases.empty
            continue
        assert len(cases) == (2 if row.validation_method == "temporal" else 5)
        assert cases.precision.ge(.30).all()
        np.testing.assert_allclose(cases.recall.mean(), row.mean_recall)
        np.testing.assert_allclose(cases.precision.mean(), row.mean_precision)
        assert cases.tp.sum() == row.pooled_tp
        assert cases.fp.sum() == row.pooled_fp

    paired = pd.read_csv(OUTPUT / "paired_cases.csv")
    for (method, architecture, fold), cases in paired.groupby(["validation_method", "architecture", "fold"]):
        rows = overall[
            overall.validation_method.eq(method) & overall.architecture.eq(architecture)
            & overall.fold.eq(fold) & overall.score_kind.eq("ensemble")
        ].set_index("feature_set")
        changes = cases.change_type.value_counts()
        assert len(cases) == rows.loc["seven", "rows"] == rows.loc["ten", "rows"]
        assert rows.loc["ten", "tp"] - rows.loc["seven", "tp"] == (
            changes.get("newly_caught_churn", 0) - changes.get("newly_missed_churn", 0)
        )
        assert rows.loc["ten", "fp"] - rows.loc["seven", "fp"] == (
            changes.get("added_false_alert", 0) - changes.get("removed_false_alert", 0)
        )
    for path in (ROOT / "debug").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
        assert all(line.rstrip() == line for line in path.read_text().splitlines()), path
    report = ROOT / "docs/debugging.md"
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        if "://" not in target:
            assert (report.parent / target.split("#")[0]).exists(), target
    assert all(line.rstrip() == line for line in report.read_text().splitlines())
    print("Verified 154 fits, hashes, shared fold assignments, seed averaging, operating metrics, paired outcomes, syntax and report links.")


if __name__ == "__main__":
    main()

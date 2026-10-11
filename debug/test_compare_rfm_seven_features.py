"""Check paired-account accounting and comparison eligibility."""

import pandas as pd
import pytest

from compare_rfm_seven_features import (
    check_baseline,
    compare_predictions,
    operating_metrics,
)


def example_predictions():
    tables = []
    for feature_set, scores in [
        ("rfm", [0.2, 0.5, 0.2, 0.8, 0.8, 0.2]),
        ("pruned", [0.8, 0.2, 0.8, 0.2, 0.8, 0.2]),
    ]:
        tables.append(pd.DataFrame({
            "fold": ["fold_2"] * 6, "StoreID": range(1, 7),
            "snapshot": ["2013-04-01"] * 6,
            "actual_churn": [1, 1, 0, 0, 1, 0],
            "n_orders_3m": [1, 0, 1, 0, 1, 0],
            "feature_set": [feature_set] * 6,
            "churn_probability": scores,
        }))
    return pd.concat(tables, ignore_index=True)


def selected_points():
    return {
        name: {"status": "selected", "threshold": 0.5}
        for name in ("rfm", "pruned")
    }


def test_pairing_classifies_all_four_changes_and_threshold_equality():
    paired = compare_predictions(example_predictions(), selected_points())
    assert paired["change_type"].tolist() == [
        "newly_caught_churn", "newly_missed_churn", "added_false_alert",
        "removed_false_alert", "unchanged", "unchanged",
    ]
    assert paired.loc[1, "rfm_alert"] == 1  # Score equals the cutoff.


def test_pairing_rejects_different_populations():
    predictions = example_predictions().drop(index=6)
    with pytest.raises(ValueError, match="different cases"):
        compare_predictions(predictions, selected_points())


def test_pairing_rejects_conflicting_outcomes():
    predictions = example_predictions()
    predictions.loc[6, "actual_churn"] = 0
    with pytest.raises(ValueError, match="different evaluation labels"):
        compare_predictions(predictions, selected_points())


def test_no_feasible_threshold_is_not_treated_as_a_selected_model():
    selections = selected_points()
    selections["pruned"] = {"status": "no_feasible_candidate"}
    with pytest.raises(ValueError, match="two feasible"):
        compare_predictions(example_predictions(), selections)
    metrics = operating_metrics(example_predictions(), selections)
    assert metrics["feature_set"].unique().tolist() == ["rfm"]


def test_activity_subgroups_reconcile_with_overall_confusion_counts():
    metrics = operating_metrics(example_predictions(), selected_points())
    for _, group in metrics.groupby("feature_set"):
        overall = group[group["subgroup"].eq("overall")].iloc[0]
        subgroups = group[group["subgroup"].ne("overall")]
        for column in ["rows", "churners", "tp", "fp", "fn", "tn"]:
            assert subgroups[column].sum() == overall[column]


def test_baseline_check_rejects_changed_scores(tmp_path):
    predictions = example_predictions()
    saved = predictions[predictions["feature_set"].eq("rfm")].copy()
    saved["candidate_id"] = "baseline"
    saved.loc[0, "churn_probability"] = 0.25
    path = tmp_path / "saved.csv"
    saved.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_baseline(predictions, path, "baseline")

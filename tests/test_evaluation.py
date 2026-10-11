import numpy as np
import pandas as pd
import pytest

from reseller_churn.evaluation import compare_cutoffs, evaluate


def examples():
    predictions = pd.DataFrame(
        {
            "StoreID": [1, 2, 3, 4],
            "snapshot": ["2013-10-01"] * 4,
            "churn_probability": [0.06, 0.55, 0.03, 0.88],
            "n_orders_3m": [0, 1, 1, 0],
        }
    )
    outcomes = predictions[["StoreID", "snapshot"]].assign(actual_churn=[1, 0, 1, 0])
    return predictions, outcomes


def test_threshold_equality_counts_and_original_roles():
    predictions, outcomes = examples()
    metrics, decisions = compare_cutoffs(
        predictions, outcomes, [0.06, 0.55, 0.88], "2014-04-01"
    )
    primary = metrics[(metrics.cutoff == 0.06) & (metrics.subgroup == "overall")].iloc[
        0
    ]
    assert [primary.tp, primary.fp, primary.fn, primary.tn] == [1, 2, 1, 0]
    assert primary.recall == 0.5
    assert primary.precision == pytest.approx(1 / 3)
    assert primary.original_evaluation_role == "primary"
    assert len(decisions) == 12
    assert (
        decisions[
            (decisions.StoreID == 1) & (decisions.cutoff == 0.06)
        ].predicted_churn.item()
        == 1
    )


def test_saved_score_equal_to_cutoff_still_alerts_after_csv_reload(tmp_path):
    boundary = 0.12345678901234568
    predictions = pd.DataFrame(
        {
            "StoreID": [1],
            "snapshot": ["2013-10-01"],
            "churn_probability": [boundary],
            "n_orders_3m": [1],
        }
    )
    outcomes = predictions[["StoreID", "snapshot"]].assign(actual_churn=1)
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    outcomes.to_csv(tmp_path / "outcomes.csv", index=False)
    evaluate(
        tmp_path / "predictions.csv",
        tmp_path / "outcomes.csv",
        tmp_path / "evaluation",
        [boundary],
        "2014-04-01",
        "inspected",
    )
    metrics = pd.read_csv(tmp_path / "evaluation/cutoff_metrics.csv")
    assert metrics[metrics.subgroup.eq("overall")].tp.item() == 1


@pytest.mark.parametrize("cutoffs", [[], [0.1, 0.1], [np.nan], [np.inf], [-0.1], [1.1]])
def test_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="Cutoffs"):
        compare_cutoffs(*examples(), cutoffs, "2014-04-01")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing_outcome",
        "extra_outcome",
        "invalid_score",
        "invalid_label",
        "invalid_count",
    ],
)
def test_invalid_prediction_or_outcome_contract(mutation):
    predictions, outcomes = examples()
    if mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif mutation == "missing_outcome":
        outcomes = outcomes.iloc[1:]
    elif mutation == "extra_outcome":
        outcomes = pd.concat([outcomes, outcomes.iloc[[0]].assign(StoreID=999)])
    elif mutation == "invalid_score":
        predictions.loc[0, "churn_probability"] = np.nan
    elif mutation == "invalid_label":
        outcomes["actual_churn"] = outcomes.actual_churn.astype(float)
        outcomes.loc[0, "actual_churn"] = 0.5
    else:
        predictions.loc[0, "n_orders_3m"] = -1
    with pytest.raises(ValueError):
        compare_cutoffs(predictions, outcomes, [0.06], "2014-04-01")


def test_mature_outcomes_required_and_original_cohort_is_inspected():
    with pytest.raises(ValueError, match="coverage"):
        compare_cutoffs(*examples(), [0.06], "2014-03-31")
    with pytest.raises(ValueError, match="already inspected"):
        compare_cutoffs(*examples(), [0.06], "2014-04-01", "independent")


def test_multiple_snapshots_are_not_pooled_and_single_class_auc_is_missing():
    predictions, outcomes = examples()
    predictions = pd.concat([predictions, predictions.assign(snapshot="2014-01-01")])
    outcomes = pd.concat(
        [
            outcomes.assign(actual_churn=0),
            outcomes.assign(snapshot="2014-01-01", actual_churn=1),
        ]
    )
    metrics, _ = compare_cutoffs(predictions, outcomes, [0, 1], "2014-07-01")
    overall = metrics[metrics.subgroup.eq("overall")]
    assert len(overall) == 4
    assert overall.rows.eq(4).all()
    assert overall.roc_auc.isna().all()
    assert (
        overall[overall.snapshot.eq(pd.Timestamp("2013-10-01"))]
        .average_precision.isna()
        .all()
    )

"""Focused checks for the diagnostic's temporal and score boundaries."""

import pandas as pd

from diagnose_april_missed_churners import (
    MATCH_FEATURES,
    SNAPSHOT,
    add_history,
    match_retained_accounts,
    score_band,
)


def test_score_band_boundaries():
    assert score_band(0.77, 0.77) == "alerted"
    assert score_band(0.67, 0.77) == "near_threshold_miss"
    assert score_band(0.669, 0.77) == "other_miss"
    assert score_band(0.30, 0.77) == "other_miss"
    assert score_band(0.299, 0.77) == "low_score_miss"


def test_history_excludes_snapshot_and_preserves_zero_order_months(tmp_path):
    orders = pd.DataFrame({
        "StoreID": [1] * 6,
        "SalesOrderID": range(6),
        "OrderDate": [
            "2012-03-31", "2012-04-01", "2012-10-01",
            "2013-01-01", "2013-04-01", "2013-04-02",
        ],
        "SubTotal": [999, 10, 20, 30, 999, 999],
    })
    path = tmp_path / "orders.csv"
    orders.to_csv(path, index=False)
    cases = pd.DataFrame({
        "StoreID": [1], "recency_days": [90],
        "n_orders_12m": [3], "revenue_12m": [60],
        "n_orders_6m": [2], "revenue_6m": [50],
        "n_orders_3m": [1], "revenue_3m": [30], "mean_gap": [91],
    })
    enriched, monthly = add_history(cases, path)
    assert len(monthly) == 12
    assert monthly["order_count"].sum() == 3
    assert monthly["revenue"].sum() == 60
    assert monthly["order_count"].eq(0).sum() == 9
    assert enriched.iloc[0]["last_order_date"] == pd.Timestamp("2013-01-01")
    assert enriched.iloc[0]["recent_revenue_ratio"] == 1.5
    assert enriched.iloc[0]["previous_3m_orders"] == 1


def test_no_previous_quarter_does_not_become_a_decline(tmp_path):
    path = tmp_path / "one_order.csv"
    pd.DataFrame({
        "StoreID": [1], "SalesOrderID": [1],
        "OrderDate": ["2013-03-30"], "SubTotal": [10],
    }).to_csv(path, index=False)
    cases = pd.DataFrame({
        "StoreID": [1], "recency_days": [2],
        "n_orders_12m": [1], "revenue_12m": [10],
        "n_orders_6m": [1], "revenue_6m": [10],
        "n_orders_3m": [1], "revenue_3m": [10], "mean_gap": [float("nan")],
    })
    enriched, _ = add_history(cases, path)
    row = enriched.iloc[0]
    assert pd.isna(row["recent_order_ratio"])
    assert pd.isna(row["recent_revenue_ratio"])
    assert not row["order_decline_25pct"]
    assert not row["revenue_decline_25pct"]
    assert row["gap_missing"]


def test_matching_uses_training_scale_and_stable_ties():
    cases = pd.DataFrame({
        "StoreID": [99, 30, 10, 20, 40],
        "churn": [1, 0, 0, 0, 0],
        "result_type": ["missed_churn"] + ["correct_non_churn"] * 4,
    })
    for column in [
        "churn_probability", *MATCH_FEATURES, "n_orders_3m", "revenue_3m",
        "recent_order_ratio", "recent_revenue_ratio", "mean_gap", "overdue_ratio",
    ]:
        cases[column] = 1.0
    cases["recency_days"] = [1.0, 2.0, 2.0, 2.0, 2.0]
    training = pd.DataFrame({
        "snapshot": pd.to_datetime(["2012-07-01", "2012-10-01"]),
        **{column: [1.0, 2.0] for column in MATCH_FEATURES},
    })
    selection = {"folds": [{
        "validation_snapshot": "2013-04-01",
        "training_snapshots": ["2012-07-01", "2012-10-01"],
    }]}
    expected = match_retained_accounts(cases, training, selection)
    assert expected["retained_StoreID"].tolist() == [10, 20, 30]
    assert expected["rfm_distance"].eq(2).all()

    # An unrelated validation row must not enter the distance normalization.
    extra = pd.DataFrame({
        "snapshot": [SNAPSHOT],
        **{column: [1e9] for column in MATCH_FEATURES},
    })
    with_validation = pd.concat([training, extra], ignore_index=True)
    actual = match_retained_accounts(cases, with_validation, selection)
    pd.testing.assert_frame_equal(actual, expected)

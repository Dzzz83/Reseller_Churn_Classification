"""Synthetic-only holdout safeguards; these tests never inspect final labels."""

import json

import numpy as np
import pandas as pd
import pytest

import run_final_testing_evaluation as final
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.evaluation.selection import evaluate_threshold_tradeoffs


def history_example():
    headers = pd.DataFrame({
        "StoreID": [1, 1, 1, 1], "SalesOrderID": [1, 2, 3, 4],
        "OrderDate": pd.to_datetime(["2012-09-30", "2012-10-01", "2013-04-01", "2013-07-01"]),
    })
    lines = pd.DataFrame({
        "SalesOrderID": [1, 2, 3, 4], "SalesOrderDetailID": [1, 2, 3, 4],
        "ProductID": [1] * 4, "OrderQty": [100, 1, 1, 2],
        "UnitPrice": [10] * 4, "UnitPriceDiscount": [.9, 0, .1, .3],
        "LineTotal": [100, 10, 9, 14],
    })
    return headers, lines


def test_final_discounts_preserve_original_windows_and_rates():
    headers, lines = history_example()
    row = final.discount_features(pd.DataFrame({"StoreID": [1]}), headers, lines, "2013-10-01").iloc[0]
    assert row.weighted_discount_3m == pytest.approx(.3)
    assert row.discount_change_3m == pytest.approx(.2)
    assert row.discounted_line_share_12m == pytest.approx(2 / 3)


def test_no_recent_orders_are_missing_not_zero_discount():
    headers, lines = history_example()
    headers = headers[headers.SalesOrderID.ne(4)]
    lines = lines[lines.SalesOrderID.ne(4)]
    row = final.discount_features(pd.DataFrame({"StoreID": [1]}), headers, lines, "2013-10-01").iloc[0]
    assert np.isnan(row.weighted_discount_3m)
    assert np.isnan(row.discount_change_3m)
    assert row.discounted_line_share_12m == .5


def test_future_orders_and_duplicate_keys_are_rejected():
    headers, lines = history_example()
    headers.loc[3, "OrderDate"] = pd.Timestamp("2013-10-01")
    with pytest.raises(ValueError, match="Future order"):
        final.discount_features(pd.DataFrame({"StoreID": [1]}), headers, lines, "2013-10-01")
    with pytest.raises(ValueError, match="Duplicate"):
        final.discount_features(pd.DataFrame({"StoreID": [1, 1]}), headers, lines, "2013-10-01")


def test_only_predeclared_development_rules_are_frozen():
    tradeoffs = evaluate_threshold_tradeoffs("example", {
        f"fold_{index}": {"target": [1, 0], "probabilities": [.9, .1]}
        for index in range(1, 6)
    })
    tradeoffs["validation_method"] = "grouped_kfold"
    tradeoffs["architecture"] = "hist_gradient_boosting"
    tradeoffs["feature_set"] = "ten"
    tradeoffs["score_kind"] = "ensemble"
    rules = final.frozen_rules(tradeoffs)
    assert [rule["minimum_development_precision"] for rule in rules] == [.2, .3, .4, .5]
    assert [rule["minimum_development_precision"] for rule in rules if rule["role"] == "primary"] == [.3]
    assert all(rule["threshold"] <= .9 for rule in rules)


def test_changed_protocol_is_rejected_before_outcome_access(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "OUTPUT", tmp_path)
    protocol = {"test_outcomes_accessed_at_freeze": False}
    (tmp_path / "frozen_protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol changed"):
        final.verify_checkpoint(protocol, {"test_outcomes_accessed": False, "protocol_sha256": "incorrect"})


def test_missing_prediction_checkpoint_never_queries_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "OUTPUT", tmp_path)
    (tmp_path / "frozen_protocol.json").write_text("{}")

    def forbidden_query(_):
        pytest.fail("Outcome query ran without an unlabeled prediction checkpoint.")

    monkeypatch.setattr(final, "query", forbidden_query)
    with pytest.raises(FileNotFoundError):
        final.evaluate()
    assert not (tmp_path / "outcome_access_started.json").exists()


def test_final_label_half_open_endpoints_with_synthetic_orders():
    orders = pd.DataFrame({
        "StoreID": [1, 2, 1, 2], "SalesOrderID": [1, 2, 3, 4],
        "OrderDate": pd.to_datetime(["2013-09-01", "2013-09-01", "2013-10-01", "2014-04-01"]),
    })
    labels = build_churn_labels(orders, PredictionWindow.from_date("2013-10-01")).set_index("StoreID")
    assert labels.loc[1, "churn"] == 0
    assert labels.loc[2, "churn"] == 1


def test_frozen_preparation_and_completed_evaluation_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="directory already exists"):
        final.prepare()
    (tmp_path / "evaluation_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="already complete"):
        final.evaluate()

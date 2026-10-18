"""Earliest-quarter share uses accepted half-open history and no outcome values."""

import json

import numpy as np
import pandas as pd
import pytest

from comparisons.basket_breadth import evaluate_fold as make_source
from comparisons.early_quarter_share import FEATURE, build_early_share
from comparisons.fixed_feature_comparison import evaluate_fixed_fold
from comparisons.grouped_precision_recall import (
    BASE_PARAMETERS,
    FEATURES,
    grouped_splits,
    synthetic_development,
)
from comparisons.recency_shape_diagnosis import quarter_counts
from comparisons.verify_quarter_history import independent_counts


def history_fixture():
    keys = pd.DataFrame({"StoreID": [1], "snapshot": pd.to_datetime(["2013-04-01"])})
    orders = pd.DataFrame(
        {
            "StoreID": 1,
            "snapshot": pd.Timestamp("2013-04-01"),
            "SalesOrderID": range(1, 7),
            "OrderDate": pd.to_datetime(
                [
                    "2012-04-01",
                    "2012-06-30",
                    "2012-07-01",
                    "2012-10-01",
                    "2013-01-01",
                    "2013-03-31",
                ]
            ),
        }
    )
    return keys, orders


def test_calendar_boundaries_and_independent_month_counts_agree():
    keys, orders = history_fixture()
    counts = quarter_counts(keys, orders)
    assert counts.filter(like="orders_quarter").iloc[0].tolist() == [2, 1, 1, 2]
    pd.testing.assert_frame_equal(counts, independent_counts(keys, orders))
    values = build_early_share(keys, orders)
    np.testing.assert_allclose(values[FEATURE], [1 / 3])
    pd.testing.assert_frame_equal(
        values, build_early_share(keys.assign(churn=1), orders)
    )
    assert list(values) == ["StoreID", "snapshot", FEATURE]


@pytest.mark.parametrize("date", ["2012-03-31", "2013-04-01"])
def test_outside_history_or_future_order_rejected(date):
    keys, orders = history_fixture()
    orders.loc[0, "OrderDate"] = pd.Timestamp(date)
    with pytest.raises(ValueError, match="history|Future"):
        build_early_share(keys, orders)


def test_duplicate_orders_invalid_snapshot_and_empty_history_rejected():
    keys, orders = history_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        build_early_share(keys, pd.concat([orders, orders.iloc[[0]]]))
    with pytest.raises(ValueError, match="Unapproved"):
        build_early_share(keys.assign(snapshot=pd.Timestamp("2013-07-01")), orders)
    with pytest.raises(ValueError, match="accepted orders"):
        build_early_share(keys, orders.iloc[:0])


def test_synthetic_nested_feature_fold_preserves_original_baseline_and_groups(tmp_path):
    data = synthetic_development()
    data["last_order_lines"] = 1.0
    data[FEATURE] = np.linspace(0, 1, len(data))
    train, held = grouped_splits(data, 5, 42)[0]
    split = dict(
        repeat=42,
        outer_fold=0,
        train_rows=train.tolist(),
        validation_rows=held.tolist(),
        train_store_ids=sorted(data.iloc[train].StoreID.unique().tolist()),
        validation_store_ids=sorted(data.iloc[held].StoreID.unique().tolist()),
    )
    baseline = make_source(data, split, tmp_path / "source")
    folder = tmp_path / "candidate"
    predicted = evaluate_fixed_fold(
        data,
        split,
        tmp_path / "source/repeat_42/fold_0",
        folder,
        dict(
            id="earliest",
            features=FEATURES + [FEATURE],
            parameters={**BASE_PARAMETERS, "max_iter": 10},
        ),
    )
    pd.testing.assert_frame_equal(
        predicted[predicted.strategy.eq("baseline")].reset_index(drop=True),
        baseline[baseline.strategy.eq("baseline")].reset_index(drop=True),
    )
    inner = pd.read_csv(folder / "inner_scores.csv")
    assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()
    assert set(inner.StoreID).isdisjoint(data.iloc[held].StoreID)
    records = json.loads((folder / "model_identities.json").read_text())
    assert len(records) == 40 and np.isfinite(predicted.score).all()

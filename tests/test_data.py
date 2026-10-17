import numpy as np
import pandas as pd
import pytest

from reseller_churn.config import FEATURES, TRAINING_SNAPSHOTS
from reseller_churn.data import (
    build_development,
    load_sources,
)
from reseller_churn.dates import parse_calendar_date
from reseller_churn.features import build_features


def test_frozen_features_and_training_boundaries(source_paths):
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    data = build_development(source)
    assert list(data.columns) == ["StoreID", "snapshot", *FEATURES, "churn"]
    assert set(data.snapshot.dt.strftime("%Y-%m-%d")) == set(TRAINING_SNAPSHOTS)
    final = build_features(source, "2013-10-01")
    assert "churn" not in final
    inactive = final[final.StoreID.eq(2)].iloc[0]
    assert inactive.n_orders_3m == 0
    assert np.isnan(inactive.weighted_discount_3m)
    assert np.isnan(inactive.discount_change_3m)
    active = final[final.StoreID.eq(3)].iloc[0]
    assert active.weighted_discount_3m == pytest.approx(0.2)
    assert active.discount_change_3m == pytest.approx(0.1)
    assert active.discount_frequency_12m == 1


def test_future_orders_do_not_enter_features(source_paths):
    before = load_sources(source_paths, "2011-01-01", "2014-04-01", "2013-10-01")
    baseline = build_features(before, "2013-10-01")
    for name in ["orders", "headers"]:
        frame = pd.read_csv(source_paths[name])
        future = frame.iloc[[0]].copy()
        future["SalesOrderID"] = 999
        future["OrderDate"] = "2013-10-01"
        pd.concat([frame, future]).to_csv(source_paths[name], index=False)
    frame = pd.read_csv(source_paths["lines"])
    future = frame.iloc[[0]].copy()
    future["SalesOrderID"] = 999
    future["SalesOrderDetailID"] = 999
    pd.concat([frame, future]).to_csv(source_paths["lines"], index=False)
    after = load_sources(source_paths, "2011-01-01", "2014-04-01", "2013-10-01")
    pd.testing.assert_frame_equal(baseline, build_features(after, "2013-10-01"))


def test_established_rejected_raw_orders_stay_excluded(source_paths):
    frame = pd.read_csv(source_paths["headers"])
    rejected = frame.iloc[[0]].assign(SalesOrderID=999, SubTotal=-1)
    pd.concat([frame, rejected]).to_csv(source_paths["headers"], index=False)
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    assert source.excluded_order_ids == (999,)
    assert not source.headers.SalesOrderID.eq(999).any()


def test_incomplete_history_or_outcome_coverage_rejected(source_paths):
    source = load_sources(source_paths, "2012-01-01", "2013-10-01", "2013-10-01")
    with pytest.raises(ValueError, match="coverage"):
        build_development(source)
    with pytest.raises(ValueError, match="coverage"):
        load_sources(source_paths, "2011-01-01", "2013-09-01", "2013-10-01")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_header",
        "missing_line",
        "wrong_revenue",
        "invalid_discount",
        "unknown_store",
        "orphan",
    ],
)
def test_invalid_source_reconciliation(source_paths, mutation):
    if mutation == "duplicate_header":
        path = source_paths["headers"]
        frame = pd.read_csv(path)
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "unknown_store":
        path = source_paths["orders"]
        frame = pd.read_csv(path)
        frame.loc[0, "StoreID"] = 999
    else:
        path = source_paths["lines"]
        frame = pd.read_csv(path)
        if mutation == "missing_line":
            frame = frame.iloc[1:]
        elif mutation == "wrong_revenue":
            frame.loc[0, "LineTotal"] = 999
        elif mutation == "orphan":
            frame.loc[0, "SalesOrderID"] = 999
        else:
            frame.loc[0, "UnitPriceDiscount"] = 1.2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")


@pytest.mark.parametrize(
    "value", ["NaT", "2013-10-01T01:00:00", "2013-10-01T00:00:00Z"]
)
def test_invalid_calendar_dates(value):
    with pytest.raises(ValueError):
        parse_calendar_date(value)


def test_eligibility_start_is_inclusive(source_paths):
    for name in ["orders", "headers"]:
        frame = pd.read_csv(source_paths[name])
        mask = frame.StoreID.eq(2) & frame.OrderDate.eq("2013-06-01")
        frame.loc[mask, "OrderDate"] = "2013-04-01"
        frame.to_csv(source_paths[name], index=False)
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    assert 2 in build_features(source, "2013-10-01").StoreID.values
    source.orders.loc[
        source.orders.StoreID.eq(2)
        & source.orders.OrderDate.eq(pd.Timestamp("2013-04-01")),
        "OrderDate",
    ] = pd.Timestamp("2013-03-31")
    assert 2 not in build_features(source, "2013-10-01").StoreID.values


def test_label_end_excluded_and_start_included(source_paths):
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    future = source.orders[source.orders.StoreID.eq(1)].iloc[[-1]].copy()
    future["SalesOrderID"] = 999
    future["OrderDate"] = pd.Timestamp("2013-01-01")
    source.orders = pd.concat([source.orders, future], ignore_index=True)
    data = build_development(source)
    july = data[data.snapshot.eq(pd.Timestamp("2012-07-01")) & data.StoreID.eq(1)]
    assert july.churn.item() == 1
    source.orders.loc[source.orders.SalesOrderID.eq(999), "OrderDate"] = pd.Timestamp(
        "2012-07-01"
    )
    data = build_development(source)
    july = data[data.snapshot.eq(pd.Timestamp("2012-07-01")) & data.StoreID.eq(1)]
    assert july.churn.item() == 0

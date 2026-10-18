"""Synthetic failures at the boundary between source files and feature rows."""

import pandas as pd
import pytest

from reseller_churn.data.preparation import build_features, load_sources
from reseller_churn.data.source_audit import audit_source_data
from reseller_churn.model import validate_features


def test_loaded_cutoff_limits_feature_coverage(source_paths):
    source = load_sources(source_paths, "2011-01-01", "2014-04-01", "2013-10-01")
    with pytest.raises(ValueError, match="loaded"):
        build_features(source, "2013-11-01")


def test_research_audit_rejects_offsetting_negative_categories(source_paths):
    orders = pd.read_csv(source_paths["orders"], parse_dates=["OrderDate"])
    stores = pd.read_csv(source_paths["stores"])
    orders.loc[0, "rev_bikes"] = -10
    orders.loc[0, "rev_accessories"] = orders.loc[0, "SubTotal"] + 10
    with pytest.raises(ValueError, match="category"):
        audit_source_data(orders, stores)


@pytest.mark.parametrize("name", ["orders", "headers"])
def test_intraday_orders_are_rejected(source_paths, name):
    frame = pd.read_csv(source_paths[name])
    frame["OrderDate"] = pd.to_datetime(frame.OrderDate)
    frame.loc[0, "OrderDate"] += pd.Timedelta(hours=12)
    frame.to_csv(source_paths[name], index=False)
    with pytest.raises(ValueError, match="midnight"):
        load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")


@pytest.mark.parametrize("mutation", ["recency", "revenue", "shares", "recent"])
def test_imported_features_must_describe_a_possible_eligible_account(source_paths, mutation):
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    features = build_features(source, "2013-10-01")
    if mutation == "recency":
        features.loc[0, "recency_days"] = 500
    elif mutation == "revenue":
        features.loc[0, "revenue_3m"] = features.loc[0, "revenue_12m"] + 100
    elif mutation == "shares":
        features.loc[0, ["share_bikes", "share_accessories"]] = [0.8, 0.8]
    else:
        features.loc[0, "n_orders_3m"] = 0
        features.loc[0, "revenue_3m"] = 100
    with pytest.raises(ValueError):
        validate_features(features)

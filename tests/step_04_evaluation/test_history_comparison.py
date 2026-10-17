"""Boundary and matching safeguards for exploratory historical comparisons."""

import numpy as np
import pandas as pd
import pytest

from comparisons.baseline_diagnosis.history_comparison import (
    CATEGORIES,
    coordinate_distances,
    history_descriptors,
    match_cases,
    similarity_coordinates,
)
from reseller_churn.config.frozen_model import FEATURES


def test_distances_preserve_missingness_and_clip():
    target = pd.Series({"a": np.nan, "b": 1.0, "c": 0.0})
    pool = pd.DataFrame({"a": [np.nan, 3.0], "b": [np.nan, 1.0], "c": [2.0, 0.1]})
    np.testing.assert_allclose(
        coordinate_distances(target, pool), [[0, 1, 1], [1, 0, 0.1]]
    )


def test_constant_and_missing_scales_are_finite():
    frame = pd.DataFrame(
        {f: [np.nan, np.nan] if "discount" in f else [0.0, 0.0] for f in FEATURES}
    )
    _, scales = similarity_coordinates(frame)
    assert all(row["scale"] == 1.0 for row in scales)
    frame.loc[0, "revenue_3m"] = -1
    with pytest.raises(ValueError, match="Negative"):
        similarity_coordinates(frame)


def test_matching_never_crosses_snapshot_and_keeps_short_pools():
    cases = pd.DataFrame({f: [0.0] * 4 for f in FEATURES})
    cases["StoreID"] = [1, 2, 3, 4]
    cases["snapshot"] = pd.to_datetime(["2013-01-01"] * 3 + ["2013-04-01"])
    cases["n_orders_3m"] = 1
    cases["churn"] = [0, 1, 1, 0]
    cases["alerts"] = 3
    matches, _, coverage = match_cases(cases)
    assert set(matches.control_StoreID) == {2, 3}
    assert coverage.loc[coverage.StoreID.eq(4), "matched_controls"].eq(0).all()
    three = matches[matches.k.eq(3)]
    assert list(three.control_StoreID) == [2, 3]
    assert coverage.loc[coverage.StoreID.eq(1), "boundary_ties"].eq(2).all()


def test_history_boundaries_and_undefined_change():
    keys = pd.DataFrame({"StoreID": [1], "snapshot": [pd.Timestamp("2013-04-01")]})
    dates = ["2012-03-31", "2012-04-01", "2013-03-31", "2013-04-01"]
    orders = pd.DataFrame(
        {
            "StoreID": 1,
            "SalesOrderID": range(1, 5),
            "OrderDate": pd.to_datetime(dates),
            "SubTotal": [100.0] * 4,
            "qty": 2,
            "n_lines": 1,
            **{c: [100.0] * 4 if c == "rev_bikes" else [0.0] * 4 for c in CATEGORIES},
        }
    )
    lines = pd.DataFrame({"SalesOrderID": range(1, 5), "ProductID": [7, 7, 8, 8]})
    profiles, trajectories = history_descriptors(keys, orders, lines)
    row = profiles.iloc[0]
    assert row.n_orders_12m == 2
    assert row.observed_span_days == 364
    assert row.category_mix_change == 0
    assert row.product_overlap == 0
    assert row.last_order_earlier_revenue_ratio == 1
    assert np.isnan(row.gap_cv) and np.isnan(row.last_gap_ratio)
    assert set(trajectories.SalesOrderID) == {2, 3}
    single, _ = history_descriptors(keys, orders.iloc[[2]], lines)
    for f in [
        "gap_cv",
        "overdue_ratio",
        "category_mix_change",
        "product_overlap",
        "last_order_earlier_revenue_ratio",
    ]:
        assert np.isnan(single.iloc[0][f])

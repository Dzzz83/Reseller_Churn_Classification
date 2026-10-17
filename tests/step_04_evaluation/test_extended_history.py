"""Extended histories remain past-only, left-truncated observations."""

import numpy as np
import pandas as pd

from comparisons.extended_history import extended_histories


def test_annual_start_is_not_older_and_future_is_excluded():
    keys = pd.DataFrame({"StoreID": [1], "snapshot": pd.to_datetime(["2013-04-01"])})
    orders = pd.DataFrame(
        {
            "StoreID": [1] * 5,
            "OrderDate": pd.to_datetime(
                ["2012-03-31", "2012-04-01", "2013-03-31", "2013-03-31", "2013-04-01"]
            ),
        }
    )
    row = extended_histories(keys, orders).iloc[0]
    assert row.all_observed_orders == 4
    assert row.orders_before_12m == 1
    assert row.all_observed_span == 365
    assert row.all_mean_gap == 182.5
    assert row.annual_overdue_ratio == 1 / 364
    assert row.newly_observable_cadence == 0


def test_no_older_observation_is_not_filled_and_single_gap_has_no_cv():
    keys = pd.DataFrame({"StoreID": [1, 2], "snapshot": pd.Timestamp("2013-04-01")})
    orders = pd.DataFrame(
        {
            "StoreID": [1, 1, 2],
            "OrderDate": pd.to_datetime(["2012-03-31", "2013-03-31", "2013-03-31"]),
        }
    )
    result = extended_histories(keys, orders).set_index("StoreID")
    assert result.loc[1, "newly_observable_cadence"] == 1
    assert result.loc[1, "all_overdue_ratio"] == 1 / 365
    assert np.isnan(result.loc[1, "all_gap_cv"])
    assert result.loc[2, "orders_before_12m"] == 0
    assert np.isnan(result.loc[2, "all_overdue_ratio"])

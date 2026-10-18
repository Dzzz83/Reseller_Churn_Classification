"""Unknown warnings stay unknown; historical ratios respect prediction boundaries."""

import numpy as np
import pandas as pd

from comparisons.hard_recent_warnings import FLAGS, add_flags, reconstruct_ratios


def test_flags_preserve_missingness_and_use_strict_boundaries():
    frame = pd.DataFrame(
        {feature: [np.nan, 1.0, 0.5, 2.0] for feature, _ in FLAGS.values()}
    )
    result = add_flags(frame)
    assert result.loc[0, list(FLAGS)].isna().all() and pd.isna(
        result.loc[0, "any_warning"]
    )
    assert result.loc[1, list(FLAGS)].eq(0).all()
    assert (
        result.loc[2, "revenue_decline"] == 1
        and result.loc[2, "overdue_vs_own_cadence"] == 0
    )
    assert (
        result.loc[3, "revenue_decline"] == 0
        and result.loc[3, "overdue_vs_own_cadence"] == 1
    )


def test_independent_ratio_reconstruction_handles_quarter_edges_and_sparse_history():
    snapshot = pd.Timestamp("2013-04-01")
    keys = pd.DataFrame(dict(StoreID=[1, 2], snapshot=snapshot))
    orders = pd.DataFrame(
        dict(
            StoreID=[1, 1, 1, 2],
            snapshot=snapshot,
            SalesOrderID=[1, 2, 3, 4],
            OrderDate=pd.to_datetime(
                ["2012-10-01", "2013-01-01", "2013-03-01", "2013-03-01"]
            ),
            SubTotal=[100.0, 50.0, 50.0, 20.0],
        )
    )
    ratios = reconstruct_ratios(keys, orders).set_index("StoreID")
    assert (
        ratios.loc[1, "revenue_ratio_3m"] == 1 and ratios.loc[1, "orders_ratio_3m"] == 2
    )
    assert ratios.loc[1, "last_order_earlier_revenue_ratio"] == 50 / 75
    assert pd.isna(ratios.loc[2, "revenue_ratio_3m"]) and pd.isna(
        ratios.loc[2, "overdue_ratio"]
    )


def test_matching_accepts_historical_feature_schema_and_keeps_all_targets():
    from comparisons import grouped_precision_recall as study
    from comparisons.hard_recent_warnings import matching

    frame = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0] for name in study.FEATURES})
    frame["StoreID"] = [1, 2, 3, 4]
    frame["snapshot"] = pd.Timestamp("2013-01-01")
    frame["churn"] = [1, 1, 0, 0]
    frame["alert"] = [False, True, False, True]
    pairs, coverage, scales = matching(frame)
    assert len(coverage) == 9 and set(coverage.StoreID) == {1}
    assert not pairs.control_StoreID.eq(1).any()
    assert len(scales) == 10 and coverage.matched_controls.gt(0).all()

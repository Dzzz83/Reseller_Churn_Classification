"""Sensitivity probes preserve clusters and do not invent calibration gains."""

import numpy as np
import pandas as pd

from comparisons.grouped_precision_recall import select_threshold
from comparisons.training_size_probe import select_training_resellers


def test_nested_training_subsets_preserve_all_reseller_snapshots():
    train = pd.DataFrame(
        {
            "StoreID": np.repeat(range(10), 2),
            "snapshot": pd.to_datetime(["2013-01-01", "2013-04-01"] * 10),
        }
    )
    half, ids50 = select_training_resellers(train, 0.5)
    larger, ids75 = select_training_resellers(train, 0.75)
    assert set(ids50).issubset(ids75)
    assert len(half) == 10 and len(larger) == 14
    assert half.groupby("StoreID").size().eq(2).all()
    pd.testing.assert_frame_equal(half, train[train.StoreID.isin(ids50)])


def test_common_strict_monotone_mapping_preserves_threshold_decisions():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    inner = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])
    outer = np.array([0.85, 0.25, 0.35, 0.001])
    cutoff = select_threshold(y, inner)["cutoff"]
    mapped = select_threshold(y, 0.7 * inner + 0.1)["cutoff"]
    np.testing.assert_array_equal(outer >= cutoff, (0.7 * outer + 0.1) >= mapped)

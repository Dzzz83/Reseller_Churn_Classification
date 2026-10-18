"""Precision-floor selection keeps ties and uses inclusive integer constraints."""

import numpy as np

from comparisons.precision_floor_october import (
    cutoff_scan,
    independent_choice,
    select_floor,
)


def test_floor_boundary_ties_and_infeasible_target():
    y = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.3, 0.3])
    scan = cutoff_scan(y, scores)
    choice = select_floor(scan, 40)
    assert choice["cutoff"] == 0.3 and choice["tp"] == 2 and choice["fp"] == 3
    assert select_floor(scan, 50) is None
    assert independent_choice(y, scores, 40) == 0.3
    assert independent_choice(y, scores, 50) is None


def test_maximum_recall_then_precision_then_higher_cutoff():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.5, 0.1])
    scan = cutoff_scan(y, scores)
    # Recall ties across several cutoffs; the best precision keeps both positives only.
    for floor in [20, 50, 100]:
        selected = select_floor(scan, floor)
        assert (
            selected["cutoff"] == 0.7
            and selected["recall"] == 1
            and selected["precision"] == 1
        )
        assert independent_choice(y, scores, floor) == 0.7

"""Outcome-derived ranking diagnostics preserve ties and isolate cutoff slack."""

import numpy as np
import pandas as pd
import pytest

from comparisons.matched_rank_burden import burden, summarize


def test_tied_scores_are_not_split_to_force_target():
    assert burden(np.array([0, 1, 1, 0]), np.array([0.9, 0.8, 0.8, 0.7]), 1) == {
        "oracle_tp": 2,
        "oracle_fp": 1,
    }
    assert burden(np.array([0, 1]), np.array([0.9, 0.8]), 0) == {
        "oracle_tp": 0,
        "oracle_fp": 0,
    }
    with pytest.raises(ValueError, match="Infeasible"):
        burden(np.array([0, 1]), np.array([0.9, 0.8]), 2)


def test_baseline_cutoff_slack_does_not_become_ranking_gain():
    base = pd.DataFrame(
        {
            "StoreID": range(4),
            "snapshot": pd.Timestamp("2013-04-01"),
            "score": [0.9, 0.8, 0.8, 0.7],
            "churn": [0, 1, 1, 0],
            "cutoff": 0.6,
            "strategy": "baseline",
        }
    )
    candidate = base.assign(cutoff=0.8, strategy="candidate")
    rows = summarize(pd.concat([base, candidate]), "strategy", 0.6)
    result = {row["policy"]: row for row in rows}
    assert result["baseline"]["actual_fp"] == 2
    assert result["candidate"]["actual_fp"] == 1
    assert result["candidate"]["oracle_fp_reduction"] == 0

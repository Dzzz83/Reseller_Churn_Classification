"""Streaming uncertainty preserves clusters, exact draws, and simulation limits."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from comparisons.bootstrap_resolution import (
    bootstrap_batches,
    failure_probability_upper,
    resolution_plan,
    simulation_audit,
)
from comparisons.matched_evaluation import paired_bootstrap


def paired_predictions():
    rows = []
    for repeat in [301, 919, 1201]:
        for store in range(1, 9):
            for snapshot, churn in [("2012-07-01", 1), ("2012-10-01", 0)]:
                for strategy in ["baseline", "candidate"]:
                    alert = (
                        store <= 6
                        if strategy == "baseline"
                        else (store != 2 if churn else store in [1, 3, 6])
                    )
                    rows.append(
                        dict(
                            StoreID=store,
                            snapshot=pd.Timestamp(snapshot),
                            churn=churn,
                            n_orders_3m=1,
                            repeat=repeat,
                            outer_fold=store % 2,
                            score=0.8 if alert else 0.2,
                            cutoff=0.5,
                            strategy=strategy,
                        )
                    )
    return pd.DataFrame(rows)


def test_streaming_draws_exactly_match_established_cluster_bootstrap():
    predictions = paired_predictions()
    expected = paired_bootstrap(
        predictions, draws=257, seed=2026, candidate_id="candidate"
    )
    batches = list(
        bootstrap_batches(predictions, "candidate", draws=257, batch_size=17)
    )
    np.testing.assert_array_equal(
        np.concatenate([row[1] for row in batches]), expected.recall_difference
    )
    np.testing.assert_array_equal(
        np.concatenate([row[2] for row in batches]), expected.fp_reduction
    )
    assert [row[0] for row in batches] == list(range(0, 257, 17))


def test_binomial_upper_bounds_match_independent_exact_scipy_interval():
    for failures in [0, 1, 30, 100]:
        expected = (
            binomtest(failures, 100, alternative="less")
            .proportion_ci(confidence_level=0.99, method="exact")
            .high
        )
        np.testing.assert_allclose(
            failure_probability_upper(failures, 100, 0.01), expected, atol=1e-12
        )


def test_fixed_extreme_statistics_reproduce_full_array_quantiles_and_failure_counts():
    predictions = paired_predictions()
    full = paired_bootstrap(predictions, draws=257, candidate_id="candidate")
    result = simulation_audit(predictions, "candidate", 0.1, draws=257, batch_size=17)
    np.testing.assert_allclose(
        result["intervals"]["recall"], np.quantile(full.recall_difference, [0.1, 0.9])
    )
    np.testing.assert_allclose(
        result["intervals"]["fp"], np.quantile(full.fp_reduction, [0.1, 0.9])
    )
    assert result["recall_negative_draws"] == full.recall_difference.lt(0).sum()
    assert result["fp_nonpositive_draws"] == full.fp_reduction.le(0).sum()
    assert result["undefined_draws"] == 0


def test_cycle_eight_tail_is_not_resolved_by_five_thousand_zero_failure_draws():
    tail = 0.000001953125
    plan = resolution_plan(tail)
    assert plan["primary_expected_tail_observations"] < 1
    assert plan["resolution_draws"] == 10240000
    assert failure_probability_upper(0, 5000, tail) > tail
    assert failure_probability_upper(0, plan["resolution_draws"], tail) < tail

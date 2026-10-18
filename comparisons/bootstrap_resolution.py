"""Prospective numerical-resolution checks for conditional reseller bootstrap tails."""

import math

import numpy as np
from scipy.stats import beta

from comparisons.baseline_diagnosis.inputs import require
from comparisons.development_reporting import validate_predictions


def failure_probability_upper(failures, draws, error_probability):
    """One-sided exact binomial upper bound on Monte Carlo failure probability."""
    require(0 <= failures <= draws and draws > 0, "Invalid Monte Carlo counts")
    require(0 < error_probability < 1, "Invalid Monte Carlo error probability")
    if failures == draws:
        return 1.0
    if failures == 0:
        return float(-np.expm1(np.log(error_probability) / draws))
    return float(beta.ppf(1 - error_probability, failures + 1, draws - failures))


def resolution_plan(tail):
    require(0 < tail < 0.5, "Invalid bootstrap tail")
    return dict(
        bootstrap_tail=tail,
        primary_draws=5000,
        primary_expected_tail_observations=5000 * tail,
        confirmation_draws=50000,
        resolution_draws=math.ceil(20 / tail),
        monte_carlo_error_probability_per_endpoint=tail,
        zero_failure_minimum_draws=math.ceil(np.log(tail) / np.log1p(-tail)),
    )


def cluster_counts(predictions, candidate_id):
    """Retain each reseller's snapshots and all repetitions in one resampling unit."""
    validate_predictions(predictions, candidate_id)
    rows = predictions.copy()
    rows["tp"] = rows.churn.eq(1) & rows.score.ge(rows.cutoff)
    rows["fp"] = rows.churn.eq(0) & rows.score.ge(rows.cutoff)
    stores = np.sort(rows.StoreID.unique())
    counts = {}
    repeats = sorted(rows.repeat.unique())
    for arm in ["baseline", candidate_id]:
        for repeat in repeats:
            part = rows[rows.strategy.eq(arm) & rows.repeat.eq(repeat)]
            counts[arm, repeat] = (
                part.groupby("StoreID")[["tp", "fp", "churn"]]
                .sum()
                .reindex(stores, fill_value=0)
                .to_numpy(dtype=float)
            )
    return stores, repeats, counts


def bootstrap_batches(predictions, candidate_id, draws, seed=2026, batch_size=5000):
    """Generate the same ordered draws as the established bootstrap, bounded in RAM."""
    require(draws > 0 and batch_size > 0, "Invalid simulation size")
    stores, repeats, counts = cluster_counts(predictions, candidate_id)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        weights = rng.multinomial(
            len(stores), np.full(len(stores), 1 / len(stores)), size=size
        )
        recall, fp = [], []
        for repeat in repeats:
            baseline = weights @ counts["baseline", repeat]
            candidate = weights @ counts[candidate_id, repeat]
            require(
                np.array_equal(baseline[:, 2], candidate[:, 2]),
                "Unmatched bootstrap populations",
            )
            recall.append(
                np.divide(
                    candidate[:, 0] - baseline[:, 0],
                    baseline[:, 2],
                    out=np.full(size, np.nan),
                    where=baseline[:, 2] > 0,
                )
            )
            fp.append(baseline[:, 1] - candidate[:, 1])
        yield start, np.mean(recall, axis=0), np.mean(fp, axis=0)


def simulation_audit(
    predictions, candidate_id, tail, *, draws=None, seed=2026, batch_size=5000
):
    """Extra gate only; never rescue failed primary or confirmation results.

    Production uses the frozen resolution_plan draw count. A smaller explicit draw
    count supports synthetic correctness checks, not an accepted research result.
    """
    plan = resolution_plan(tail)
    draws = plan["resolution_draws"] if draws is None else draws
    require(draws >= 2, "At least two simulation draws required")
    keep = int(math.ceil((draws - 1) * tail)) + 2
    extremes = {
        name: np.array([], dtype=float)
        for name in ["recall_low", "recall_high", "fp_low", "fp_high"]
    }
    recall_failures, fp_failures, undefined = 0, 0, 0
    for _, recall, fp in bootstrap_batches(
        predictions, candidate_id, draws, seed, batch_size
    ):
        undefined += int((~np.isfinite(recall) | ~np.isfinite(fp)).sum())
        recall_failures += int((recall < 0).sum())
        fp_failures += int((fp <= 0).sum())
        for name, values in [("recall", recall), ("fp", fp)]:
            finite = values[np.isfinite(values)]
            for suffix, high in [("low", False), ("high", True)]:
                key = f"{name}_{suffix}"
                pool = np.concatenate([extremes[key], finite])
                if len(pool) > keep:
                    point = len(pool) - keep if high else keep - 1
                    pool = (
                        np.partition(pool, point)[-keep:]
                        if high
                        else np.partition(pool, point)[:keep]
                    )
                extremes[key] = pool
    recall_upper = failure_probability_upper(recall_failures, draws, tail)
    fp_upper = failure_probability_upper(fp_failures, draws, tail)
    # Undefined draws cannot be silently omitted or interpreted as successes.
    passed = undefined == 0 and recall_upper <= tail and fp_upper <= tail
    quantiles = {}
    if undefined == 0:
        for name in ["recall", "fp"]:
            position = (draws - 1) * tail
            low = np.sort(extremes[f"{name}_low"])
            high = np.sort(extremes[f"{name}_high"])[::-1]
            index, fraction = math.floor(position), position - math.floor(position)
            quantiles[name] = [
                float(low[index] + fraction * (low[index + 1] - low[index])),
                float(high[index] + fraction * (high[index + 1] - high[index])),
            ]
    return dict(
        plan=plan,
        draws=draws,
        seed=seed,
        passed=bool(passed),
        recall_negative_draws=recall_failures,
        fp_nonpositive_draws=fp_failures,
        undefined_draws=undefined,
        recall_failure_probability_upper=recall_upper,
        fp_failure_probability_upper=fp_upper,
        intervals=quantiles,
        extreme_order_statistics={
            name: sorted(values.tolist()) for name, values in extremes.items()
        },
        limitation="Bounds quantify Monte Carlo error conditional on the empirical bootstrap, not model or adaptive-research coverage",
    )

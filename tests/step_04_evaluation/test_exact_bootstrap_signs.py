"""Integer convolution equals exhaustive ordered cluster resampling on small cases."""

from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd
import pytest

from comparisons.exact_bootstrap_signs import exact_sum_tail, sign_audit


@pytest.mark.parametrize(
    "values", [[-1, 0, 2], [-3, -3, 3, 3], [0, 0, 0], [1, 2], [-1]]
)
@pytest.mark.parametrize("nonpositive", [False, True])
def test_exact_polynomial_counts_equal_exhaustive_resamples(values, nonpositive):
    result = exact_sum_tail(values, nonpositive=nonpositive)
    sums = [sum(sample) for sample in product(values, repeat=len(values))]
    count = sum(value <= 0 if nonpositive else value < 0 for value in sums)
    assert result["numerator"] == count
    assert result["denominator"] == len(values) ** len(values)
    assert Fraction(result["numerator"], result["denominator"]) == Fraction(
        count, len(sums)
    )


def paired_predictions():
    rows = []
    for repeat in [301, 919, 1201]:
        for store in [1, 2, 3]:
            for day, churn in [("2012-07-01", 1), ("2012-10-01", 0)]:
                for arm in ["baseline", "candidate"]:
                    alert = (
                        store <= 2
                        if arm == "baseline"
                        else (store != 1 if churn else store == 2)
                    )
                    rows.append(
                        dict(
                            StoreID=store,
                            snapshot=pd.Timestamp(day),
                            churn=churn,
                            n_orders_3m=1,
                            repeat=repeat,
                            outer_fold=store % 2,
                            score=0.8 if alert else 0.2,
                            cutoff=0.5,
                            strategy=arm,
                        )
                    )
    return pd.DataFrame(rows)


def test_paired_sign_reduction_matches_direct_recall_and_fp_calculation():
    predictions = paired_predictions()
    recall_failures, fp_failures = 0, 0
    for sample in product([1, 2, 3], repeat=3):
        recall_changes, fp_reductions = [], []
        for repeat in [301, 919, 1201]:
            results = {}
            for arm in ["baseline", "candidate"]:
                part = predictions[
                    predictions.repeat.eq(repeat) & predictions.strategy.eq(arm)
                ]
                selected = pd.concat([part[part.StoreID.eq(store)] for store in sample])
                alert, y = selected.score.ge(selected.cutoff), selected.churn.eq(1)
                results[arm] = (
                    int((alert & y).sum()) / int(y.sum()),
                    int((alert & ~y).sum()),
                )
            recall_changes.append(results["candidate"][0] - results["baseline"][0])
            fp_reductions.append(results["baseline"][1] - results["candidate"][1])
        recall_failures += np.mean(recall_changes) < 0
        fp_failures += np.mean(fp_reductions) <= 0
    audit = sign_audit(predictions, "candidate", 0.01)
    assert int(audit["recall"]["numerator"]) == recall_failures
    assert int(audit["fp"]["numerator"]) == fp_failures
    assert int(audit["recall"]["denominator"]) == 27
    assert audit["undefined_numerator"] == "0"


def test_undefined_churn_denominator_is_conservatively_adverse():
    predictions = paired_predictions()
    predictions.loc[predictions.StoreID.eq(1), "churn"] = 0
    audit = sign_audit(predictions, "candidate", 0.01)
    assert audit["undefined_numerator"] == "1"
    assert audit["undefined_probability"] == 1 / 27
    assert not audit["recall"]["conditional_sign_gate_passed"]


@pytest.mark.parametrize("values", [[], [np.nan], [0.5]])
def test_noninteger_or_invalid_cluster_statistics_are_rejected(values):
    with pytest.raises(ValueError):
        exact_sum_tail(values, nonpositive=True)

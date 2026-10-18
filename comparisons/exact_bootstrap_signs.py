"""Exact empirical reseller-bootstrap sign probabilities by integer convolution.

This verifies numerical tails, not independence, retraining, or future coverage.
It does not replace a historical experiment's frozen acceptance procedure.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.development_reporting import validate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

KEYS = ["StoreID", "snapshot", "repeat", "outer_fold", "churn", "n_orders_3m"]


def exact_sum_tail(values, *, nonpositive):
    """Count all n**n ordered reseller resamples with sum<0 or sum<=0.

    Polynomial coefficients are arbitrary-precision Python integers. No FFT,
    floating-point convolution, asymptotic tail approximation or simulation.
    """
    values = np.asarray(values)
    require(
        values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
        "Invalid cluster values",
    )
    require(np.equal(values, np.floor(values)).all(), "Expected integer cluster deltas")
    values = values.astype(np.int64)
    unique, frequencies = np.unique(values, return_counts=True)
    n = len(values)
    denominator = n**n
    minimum = int(unique[0])
    step = reduce(gcd, [int(x - minimum) for x in unique], 0)
    if step == 0:
        coefficients = np.array([denominator], dtype=object)
    else:
        polynomial = np.zeros((int(unique[-1]) - minimum) // step + 1, dtype=object)
        for value, count in zip(unique, frequencies):
            polynomial[(int(value) - minimum) // step] = int(count)
        coefficients = np.array([1], dtype=object)
        for _ in range(n):
            coefficients = np.convolve(coefficients, polynomial)
    require(
        sum(coefficients) == denominator, "Exact bootstrap probability mass changed"
    )
    threshold = 0 if nonpositive else -1
    if step == 0:
        numerator = denominator if n * minimum <= threshold else 0
    else:
        last = (threshold - n * minimum) // step
        numerator = (
            0
            if last < 0
            else int(sum(coefficients[: min(last + 1, len(coefficients))]))
        )
    digest = hashlib.sha256()
    for coefficient in coefficients:
        digest.update(str(int(coefficient)).encode("ascii") + b"\n")
    return dict(
        numerator=numerator,
        denominator=denominator,
        probability=float(Fraction(numerator, denominator)),
        clusters=n,
        minimum_sum=n * minimum,
        lattice_step=step,
        states=len(coefficients),
        cluster_values=unique.tolist(),
        cluster_frequencies=frequencies.tolist(),
        coefficient_sha256=digest.hexdigest(),
        coefficient_encoding="decimal-integer-newline-v1",
    )


def paired_cluster_changes(predictions, candidate_id):
    validate_predictions(predictions, candidate_id)
    baseline = predictions[predictions.strategy.eq("baseline")]
    candidate = predictions[predictions.strategy.eq(candidate_id)]
    paired = baseline.merge(
        candidate, on=KEYS, suffixes=("_base", "_new"), validate="one_to_one"
    )
    a = paired.score_base.ge(paired.cutoff_base).astype(int)
    b = paired.score_new.ge(paired.cutoff_new).astype(int)
    paired["tp_change"] = (b - a) * paired.churn
    paired["fp_reduction"] = (a - b) * (1 - paired.churn)
    labels = paired.groupby(["StoreID", "repeat"]).churn.sum().unstack("repeat")
    require(
        labels.notna().all().all() and labels.nunique(axis=1).eq(1).all(),
        "Recall denominators differ across repeats; sign reduction is invalid",
    )
    changes = (
        paired.groupby("StoreID")[["tp_change", "fp_reduction"]].sum().sort_index()
    )
    changes["churn_cases"] = labels.iloc[:, 0].reindex(changes.index)
    return changes


def sign_audit(predictions, candidate_id, tail):
    """Exact conditional sign checks, with undefined recall conservatively adverse."""
    changes = paired_cluster_changes(predictions, candidate_id)
    recall = exact_sum_tail(changes.tp_change.to_numpy(), nonpositive=False)
    fp = exact_sum_tail(changes.fp_reduction.to_numpy(), nonpositive=True)
    n = len(changes)
    undefined_numerator = int(changes.churn_cases.eq(0).sum()) ** n
    denominator = n**n
    target = Fraction(str(tail))
    require(0 < target < Fraction(1, 2), "Invalid tail")
    outputs = {}
    for name, result in [("recall", recall), ("fp", fp)]:
        # For recall the events are disjoint. For FP this union bound can overcount,
        # so it is conservative; it can never rescue a failed sign requirement.
        upper = min(denominator, result["numerator"] + undefined_numerator)
        passed = upper * target.denominator < target.numerator * denominator
        outputs[name] = {
            **result,
            "numerator": str(result["numerator"]),
            "denominator": str(denominator),
            "undefined_adverse_upper_numerator": str(upper),
            "undefined_adverse_probability_upper": float(Fraction(upper, denominator)),
            "conditional_sign_gate_passed": bool(passed),
        }
    return dict(
        status="exact_empirical_bootstrap",
        tail=float(target),
        clusters=n,
        repeat_denominators_verified_identical=True,
        undefined_numerator=str(undefined_numerator),
        undefined_probability=float(Fraction(undefined_numerator, denominator)),
        recall=outputs["recall"],
        fp=outputs["fp"],
        conditional_sign_gates_passed=all(
            item["conditional_sign_gate_passed"] for item in outputs.values()
        ),
        historical_acceptance_changed=False,
        limitations="Exact counting for the fitted empirical cluster bootstrap only; omits retraining, adaptive search, calendar dependence and future populations",
    )


def run(source, candidate_id, tail, output):
    verify_manifest(source)
    protocol = json.loads((source / "protocol.json").read_text())
    require(
        float(tail) == protocol["interval_tail_probability"],
        "Audit tail differs from frozen protocol",
    )
    predictions = read_csv(source / "validation_predictions.csv")
    audit = sign_audit(predictions, candidate_id, tail)
    out = new_output_directory(output)
    write_json(out / "exact_sign_audit.json", audit)
    paired_cluster_changes(predictions, candidate_id).to_csv(
        out / "cluster_changes.csv"
    )
    (out / "report.md").write_text(
        "# Exact conditional bootstrap sign audit\n\n"
        f"{audit['clusters']} reseller clusters, all snapshots and repeated predictions carried together. "
        "Integer polynomial multiplication counts all ordered reseller resamples exactly. No model fits, "
        "new customer evidence, cutoff changes or historical acceptance changes.\n\n"
        f"Recall-negative probability: {audit['recall']['probability']:.12g}. "
        f"FP-nonpositive probability: {audit['fp']['probability']:.12g}. "
        f"Frozen tail: {tail:.12g}. Conservative undefined-recall probability: {audit['undefined_probability']:.12g}.\n\n"
        "Every repeat has the same resampled churn denominator because the matched cases/labels are identical. "
        "Thus the sign of mean recall difference is the sign of the sum of repeat TP differences. "
        "The FP-reduction sign is additive directly. The full recall interval magnitude is not computed here. "
        "Numerators, common denominator, cluster histogram and coefficient hashes are retained. \n\n"
        "Exact arithmetic addresses simulation error only. Adaptive research, refitting, calendar-wide "
        "dependence and future population changes remain outside this empirical bootstrap. Earlier frozen "
        "Monte Carlo protocols and failed acceptance decisions remain unchanged.\n\n"
        "Primary context: https://arxiv.org/abs/2107.08019 and "
        "https://optimization-online.org/wp-content/uploads/2017/09/6211.pdf .\n"
    )
    write_manifest(
        out,
        dict(
            models_fitted=0,
            historical_acceptance_changed=False,
            inputs=input_identity(
                {"source": source / "manifest.json", "script": Path(__file__)}
            ),
        ),
    )
    print({name: audit[name]["probability"] for name in ["recall", "fp"]})
    print("Exact conditional sign gates:", audit["conditional_sign_gates_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tail", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.candidate, args.tail, args.output)

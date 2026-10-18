"""Fixed eight-cell period/class-balanced HGB loss on the original ten features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.fixed_feature_comparison import run_fixed_hgb
from comparisons.reseller_weighting import fit_weighted
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "period_class_weighted"
TAIL = 0.0000078125


def period_weights(train):
    require(
        set(train.snapshot.dt.strftime("%Y-%m-%d")) == set(study.SNAPSHOTS),
        "All four approved fitting periods required",
    )
    counts = train.groupby(["snapshot", "churn"]).size()
    require(
        len(counts) == 8 and train.churn.isin([0, 1]).all(),
        "Both fitting outcomes required in every period",
    )
    return np.array(
        [
            len(train) / (8 * counts.loc[(row.snapshot, row.churn)])
            for row in train[["snapshot", "churn"]].itertuples(index=False)
        ]
    )


def fit_period_weighted(train, validation_features, spec):
    return fit_weighted(
        train, validation_features, spec, weight_function=period_weights
    )


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(history_source)
    data = read_csv(source / "development_input.csv")
    history = read_csv(history_source / "case_histories.csv")
    profiles = history[study.KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    spec = dict(
        id=CANDIDATE,
        features=study.FEATURES,
        parameters=study.BASE_PARAMETERS,
        description="Original HGB with fixed equal period/class fitting cell weights",
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/period_weighting_protocol.md",
        TAIL,
        dict(
            history=history_source / "manifest.json",
            entrypoint=Path(__file__),
            weighted_fitter=ROOT / "comparisons/reseller_weighting.py",
        ),
        period_limits=LIMITS,
        fit_function=fit_period_weighted,
        post_assessment=intersection_gate,
    )
    # Counts/reporting are provided by the shared assessment; no separate quality claim.
    print(
        pd.read_csv(output / "paired_repeat_metrics.csv")[
            ["repeat", "tp_baseline", "tp_candidate", "fp_baseline", "fp_candidate"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

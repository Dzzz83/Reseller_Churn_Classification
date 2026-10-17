"""One prespecified last-basket paid-unit-revenue feature added to cadence HGB."""

import argparse
from pathlib import Path

import numpy as np

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.fixed_feature_comparison import run_fixed_hgb
from reseller_churn.artifacts import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(history_source)
    data = read_csv(source / "development_input.csv")
    feature = "last_order_unit_revenue"
    histories = read_csv(history_source / "case_histories.csv")
    data = data.merge(histories[KEYS + [feature]], on=KEYS, validate="one_to_one")
    require(
        len(data) == 1375
        and np.isfinite(data[feature]).all()
        and data[feature].gt(0).all(),
        "Invalid paid unit revenue",
    )
    profiles = histories[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    spec = dict(
        id="cadence_unit_revenue",
        description="Fixed cadence plus latest paid unit revenue",
        features=study.FEATURES + ["overdue_ratio", "gap_cv", feature],
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/cadence_unit_revenue_protocol.md",
        0.05 / 56,
        dict(history=history_source / "manifest.json", entrypoint=Path(__file__)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

"""One historical purchase-shape feature added to the unchanged original HGB."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.additional_uncertainty import additional_confirmation
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.fixed_feature_comparison import run_fixed_hgb
from comparisons.recency_shape_diagnosis import quarter_counts
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    input_identity,
    verify_manifest,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
FEATURE = "early_quarter_order_share"
CANDIDATE = "early_quarter_share"
TAIL = 0.000001953125


def build_early_share(keys, trajectories):
    """Use each case's accepted historical orders; labels are not inputs."""
    cases = keys[KEYS].copy()
    require(
        not cases.duplicated(KEYS).any() and not cases[KEYS].isna().any().any(),
        "Invalid case keys",
    )
    require(
        set(cases.snapshot.dt.strftime("%Y-%m-%d")).issubset(study.SNAPSHOTS),
        "Unapproved snapshot",
    )
    counts = quarter_counts(cases, trajectories)
    denominator = counts[[f"orders_quarter_{i}" for i in range(1, 5)]].sum(axis=1)
    require(denominator.gt(0).all(), "Eligible histories require accepted orders")
    counts[FEATURE] = counts.orders_quarter_1 / denominator
    require(counts[FEATURE].between(0, 1).all(), "Invalid historical-quarter share")
    return counts[KEYS + [FEATURE]]


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(history_source)
    data = read_csv(source / "development_input.csv")
    histories = read_csv(history_source / "case_histories.csv")
    trajectories = read_csv(history_source / "order_trajectories.csv")
    trajectories["OrderDate"] = pd.to_datetime(trajectories.OrderDate)
    feature = build_early_share(data[KEYS], trajectories)
    feature_audit = ROOT / "results/quarter-history-independent-audit-20261018"
    verify_manifest(feature_audit)
    independent = read_csv(feature_audit / "independent_features.csv")
    checked = feature.merge(
        independent[KEYS + [FEATURE]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("_built", "_independent"),
    )
    require(
        len(checked) == len(feature), "Independent feature audit population differs"
    )
    np.testing.assert_array_equal(
        checked[FEATURE + "_built"], checked[FEATURE + "_independent"]
    )
    data = data.merge(feature, on=KEYS, validate="one_to_one")
    require(
        len(data) == 1375 and np.isfinite(data[FEATURE]).all(),
        "Feature population changed",
    )
    profiles = histories[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    spec = dict(
        id=CANDIDATE,
        description="Original HGB plus earliest historical-quarter order share",
        features=study.FEATURES + [FEATURE],
        parameters=dict(study.BASE_PARAMETERS),
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/early_quarter_share_protocol.md",
        TAIL,
        dict(
            history=history_source / "manifest.json",
            feature_audit=feature_audit / "manifest.json",
            entrypoint=Path(__file__),
            feature_builder=ROOT / "comparisons/recency_shape_diagnosis.py",
            resolution=ROOT / "comparisons/bootstrap_resolution.py",
            confirmation=ROOT / "comparisons/additional_uncertainty.py",
            intersection_check=ROOT / "comparisons/retained_calibration.py",
        ),
        period_limits=LIMITS,
        post_assessment=intersection_gate,
    )
    final = additional_confirmation(output, CANDIDATE, TAIL)
    bindings = {
        "primary_manifest": output / "manifest.json",
        "primary_decision": output / "decision.json",
        "final_decision": output / "final_decision.json",
        "resolution_plan": output / "resolution_plan.json",
    }
    if (output / "confirmation/manifest.json").is_file():
        bindings["confirmation"] = output / "confirmation/manifest.json"
    write_json(
        output / "procedure_completion.json",
        {
            "accepted": final["accepted"],
            "artifacts": input_identity(bindings),
        },
    )
    print("Complete procedure accepted:", final["accepted"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

"""Independent calendar-month reconstruction of approved historical-quarter features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def independent_counts(keys, trajectories):
    """Quarter-start snapshots permit a separate integer-calendar formulation."""
    expected = keys[KEYS].copy()
    for quarter in range(1, 5):
        expected[f"orders_quarter_{quarter}"] = 0
    require(
        not trajectories.duplicated(KEYS + ["SalesOrderID"]).any(),
        "Duplicate accepted order",
    )
    require(
        trajectories.OrderDate.lt(trajectories.snapshot).all(),
        "Future trajectory order",
    )
    lookup = expected.set_index(KEYS).index
    positions = {key: index for index, key in enumerate(lookup)}
    for row in trajectories.itertuples(index=False):
        require(row.snapshot.day == 1, "Expected calendar-quarter snapshot")
        month_age = (
            (row.snapshot.year - row.OrderDate.year) * 12
            + row.snapshot.month
            - row.OrderDate.month
        )
        require(1 <= month_age <= 12, "Trajectory outside established history")
        quarter = 4 - (month_age - 1) // 3
        key = (row.StoreID, row.snapshot)
        if key in positions:
            index = positions[key]
            expected.loc[index, f"orders_quarter_{quarter}"] += 1
    return expected


def verify(output):
    source = ROOT / "results/persistent-error-investigation-20261016"
    diagnosis = ROOT / "results/recency-shape-diagnosis-20261018"
    verify_manifest(source)
    verify_manifest(diagnosis)
    trajectories = read_csv(source / "order_trajectories.csv")
    trajectories["OrderDate"] = pd.to_datetime(trajectories.OrderDate)
    recorded = read_csv(diagnosis / "quarter_counts.csv")
    expected = independent_counts(recorded[KEYS], trajectories)
    columns = [f"orders_quarter_{q}" for q in range(1, 5)]
    aligned = recorded.merge(
        expected, on=KEYS, validate="one_to_one", suffixes=("_saved", "_independent")
    )
    for column in columns:
        np.testing.assert_array_equal(
            aligned[column + "_saved"], aligned[column + "_independent"]
        )
    denominator = expected[columns].sum(axis=1)
    np.testing.assert_array_equal(denominator, recorded.n_orders_12m)
    np.testing.assert_array_equal(expected.orders_quarter_4, recorded.n_orders_3m)
    expected["early_quarter_order_share"] = expected.orders_quarter_1 / denominator
    out = new_output_directory(output)
    expected.to_csv(out / "independent_features.csv", index=False)
    result = dict(
        status="passed",
        cases=len(recorded),
        quarter_values=4 * len(recorded),
        models_fitted=0,
    )
    write_json(out / "verification.json", result)
    write_manifest(
        out,
        dict(
            inputs=input_identity(
                {
                    "trajectory_source": source / "manifest.json",
                    "diagnosis": diagnosis / "manifest.json",
                    "verifier": Path(__file__),
                }
            ),
            **result,
        ),
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)

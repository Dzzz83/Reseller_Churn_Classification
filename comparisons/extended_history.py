"""No-fit matched screen of observed purchases preceding the annual window."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.history_comparison import (
    matched_differences,
    summarize_differences,
)
from comparisons.baseline_diagnosis.inputs import read_csv, require, verify_baseline
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)
from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]
DESCRIPTORS = [
    "all_observed_orders",
    "all_observed_span",
    "orders_before_12m",
    "all_mean_gap",
    "all_gap_cv",
    "all_overdue_ratio",
    "all_last_gap_ratio",
    "annual_overdue_ratio",
    "newly_observable_cadence",
]


def extended_histories(keys, orders):
    require(not keys.duplicated(KEYS).any(), "Duplicate case")
    require(
        set(keys.snapshot.dt.strftime("%Y-%m-%d")).issubset(TRAINING_SNAPSHOTS),
        "Nondevelopment snapshot",
    )
    rows = []
    for snapshot, cases in keys.groupby("snapshot"):
        history = orders[orders.OrderDate.lt(snapshot)]
        for store in cases.StoreID:
            past = history[history.StoreID.eq(store)]
            require(len(past) > 0, "Missing historical purchases")
            dates = past.OrderDate.drop_duplicates().sort_values()
            gaps = dates.diff().dt.days.dropna()
            annual = dates[dates.ge(snapshot - pd.DateOffset(months=12))]
            annual_gaps = annual.diff().dt.days.dropna()
            recency = (snapshot - dates.max()).days
            mean = gaps.mean()
            rows.append(
                dict(
                    StoreID=store,
                    snapshot=snapshot,
                    all_observed_orders=len(past),
                    all_observed_span=(dates.max() - dates.min()).days,
                    orders_before_12m=int(
                        past.OrderDate.lt(snapshot - pd.DateOffset(months=12)).sum()
                    ),
                    all_mean_gap=mean,
                    all_gap_cv=gaps.std(ddof=0) / mean if len(gaps) >= 2 else np.nan,
                    all_overdue_ratio=recency / mean if mean > 0 else np.nan,
                    all_last_gap_ratio=gaps.iloc[-1] / gaps.iloc[:-1].mean()
                    if len(gaps) >= 2
                    else np.nan,
                    annual_overdue_ratio=recency / annual_gaps.mean()
                    if len(annual_gaps)
                    else np.nan,
                    newly_observable_cadence=int(
                        len(gaps) > 0 and len(annual_gaps) == 0
                    ),
                )
            )
    return pd.DataFrame(rows)


def run(output):
    orders_path = (
        ROOT / "results/research-provenance-20261016/orders_clean_historical_bytes.csv"
    )
    data, _, _ = verify_baseline(
        ROOT / "results/repeated-grouped-precision", orders_path
    )
    source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    orders = pd.read_csv(orders_path, parse_dates=["OrderDate"])
    orders = orders[orders.OrderDate.lt(data.snapshot.max())]
    profiles = extended_histories(data[KEYS], orders)
    prior = read_csv(source / "case_histories.csv")
    check = profiles.merge(
        prior[KEYS + ["overdue_ratio"]], on=KEYS, validate="one_to_one"
    )
    np.testing.assert_allclose(
        check.annual_overdue_ratio, check.overdue_ratio, equal_nan=True, rtol=1e-12
    )
    differences = matched_differences(
        profiles, read_csv(source / "matches.csv"), DESCRIPTORS
    )
    summary = summarize_differences(differences)
    out = new_output_directory(output)
    for name, table in [
        ("case_histories", profiles),
        ("matched_differences", differences),
        ("summary", summary),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    coverage = profiles.groupby("snapshot").agg(
        cases=("StoreID", "size"),
        older_observed=("orders_before_12m", lambda x: int(x.gt(0).sum())),
        newly_observable=("newly_observable_cadence", "sum"),
    )
    coverage.to_csv(out / "coverage.csv")
    write_manifest(
        out,
        dict(
            stage="extended_observed_history",
            models_fitted=0,
            cutoffs_selected=0,
            october_outcomes_accessed=False,
            source_start="2011-05-31 (completeness assumed)",
            inputs=input_identity(
                dict(
                    orders=orders_path,
                    matches=source / "matches.csv",
                    protocol=ROOT / "comparisons/persistent_error_investigation.md",
                )
            ),
            implementation=input_identity(
                dict(
                    runner=Path(__file__),
                    summary=ROOT
                    / "comparisons/baseline_diagnosis/history_comparison.py",
                )
            ),
        ),
    )
    print(coverage.to_string())
    print(
        summary[summary.k.eq(3) & summary.feature.eq("all_overdue_ratio")][
            [
                "comparison",
                "scope",
                "snapshot",
                "targets",
                "paired_observed",
                "mean_difference",
                "median_difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

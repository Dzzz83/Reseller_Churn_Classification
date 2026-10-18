"""Development-only recency/history-shape summaries; no estimator or cutoff fitting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.breadth_reporting import markdown_table
from comparisons.development_checks import validate_development_cases
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def recency_summaries(data):
    frame = data.copy()
    frame["recency_band"] = pd.cut(frame.recency_days, [-1, 30, 60, 90, 120, 150, 184])
    frame["activity"] = np.where(
        frame.n_orders_3m.gt(0), "recent_purchase", "no_recent_purchase"
    )
    tables = []
    for dimension, columns in [
        ("snapshot", ["snapshot", "recency_band"]),
        ("exact_annual_orders", ["snapshot", "n_orders_12m", "recency_band"]),
    ]:
        grouped = (
            frame.groupby(columns, observed=False)
            .churn.agg(rows="size", churners="sum")
            .reset_index()
        )
        grouped["churn_rate"] = grouped.churners / grouped.rows.where(
            grouped.rows.gt(0)
        )
        grouped["small_sample"] = (
            grouped.rows.lt(30)
            | grouped.churners.lt(10)
            | (grouped.rows - grouped.churners).lt(10)
        )
        grouped["dimension"] = dimension
        tables.append(grouped)
    return tables


def quarter_counts(keys, trajectories):
    """Count accepted orders in four half-open historical quarters per case."""
    require(
        not trajectories.duplicated(KEYS + ["SalesOrderID"]).any(),
        "Duplicate trajectory order",
    )
    require(
        trajectories.OrderDate.lt(trajectories.snapshot).all(),
        "Future order in history",
    )
    results = []
    for snapshot, cases in keys.groupby("snapshot"):
        purchases = trajectories[trajectories.snapshot.eq(snapshot)]
        require(
            purchases.OrderDate.ge(snapshot - pd.DateOffset(months=12)).all(),
            "Order predates allowed history",
        )
        rows = cases[KEYS].copy()
        for quarter in range(4):
            start = snapshot - pd.DateOffset(months=12 - 3 * quarter)
            end = snapshot - pd.DateOffset(months=9 - 3 * quarter)
            selected = purchases[
                purchases.OrderDate.ge(start) & purchases.OrderDate.lt(end)
            ]
            count = selected.groupby("StoreID").SalesOrderID.nunique()
            rows[f"orders_quarter_{quarter + 1}"] = (
                rows.StoreID.map(count).fillna(0).astype(int)
            )
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(history_source)
    data = read_csv(source / "development_input.csv")
    validate_development_cases(data)
    histories = read_csv(history_source / "case_histories.csv")
    if "n_orders_12m" not in data:
        data = data.merge(
            histories[KEYS + ["n_orders_12m"]], on=KEYS, validate="one_to_one"
        )
    trajectory = read_csv(history_source / "order_trajectories.csv")
    trajectory["OrderDate"] = pd.to_datetime(trajectory.OrderDate)
    counts = quarter_counts(data[KEYS], trajectory)
    joined = data.merge(counts, on=KEYS, validate="one_to_one")
    require(len(joined) == 1375, "Case population changed")
    quarter_columns = [f"orders_quarter_{i}" for i in range(1, 5)]
    np.testing.assert_array_equal(
        joined[quarter_columns].sum(axis=1), joined.n_orders_12m
    )
    np.testing.assert_array_equal(joined.orders_quarter_4, joined.n_orders_3m)
    out = new_output_directory(output)
    broad, conditional = recency_summaries(joined)
    broad.to_csv(out / "recency_by_snapshot.csv", index=False)
    conditional.to_csv(out / "recency_by_exact_count.csv", index=False)
    joined[KEYS + ["churn", "n_orders_3m", "n_orders_12m", *quarter_columns]].to_csv(
        out / "quarter_counts.csv", index=False
    )
    (out / "report.md").write_text(
        "# Recency shape and historical-quarter diagnosis\n\n"
        "No classifier, calibrator, cutoff or candidate was fitted. All 1,375 development cases retained.\n\n"
        + markdown_table(broad.drop(columns=["dimension"]))
        + "\n\nThese marginal rates do not establish partial dependence or causality. Exact annual-count strata "
        "are separately retained, with sparse/undefined cells explicit. BG/NBD P(alive) conditions on "
        "frequency and observation age and assumes a particular purchase/dropout process; it is not "
        "this six-month no-purchase outcome. This evidence does not justify treating global recency "
        "monotonicity as an established law. No hard monotonic constraint is selected here.\n\n"
        "Four historical quarter counts are reconstructed solely from saved accepted pre-snapshot "
        "trajectories, using [t-12m,t-9m), [t-9m,t-6m), [t-6m,t-3m), [t-3m,t). Their sum exactly "
        "matches annual order count and the final quarter matches n_orders_3m for every case. "
        "This is a provenance/representation check, not a feature-performance result or seasonality proof.\n\n"
        "Primary references: https://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf and "
        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html .\n"
    )
    write_manifest(
        out,
        dict(
            stage="development_recency_shape",
            models_fitted=0,
            inputs=input_identity(
                {
                    "development": source / "manifest.json",
                    "history": history_source / "manifest.json",
                    "script": Path(__file__),
                }
            ),
            canonical_cases=1375,
            quarter_reconciliations=1375,
        ),
    )
    print("Reconciled four historical quarters for all 1,375 cases; no model fitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

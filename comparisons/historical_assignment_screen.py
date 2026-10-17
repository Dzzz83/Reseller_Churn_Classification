"""Exploratory categorical order-assignment histories; no models or cutoffs."""

import argparse
from pathlib import Path

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)
from reseller_churn.data.prediction_window import PredictionWindow

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def run(output):
    source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    orders_path = (
        ROOT / "results/research-provenance-20261016/orders_clean_historical_bytes.csv"
    )
    orders = pd.read_csv(orders_path, parse_dates=["OrderDate"])
    cases = read_csv(source / "case_histories.csv")
    profiles = []
    for snapshot, cohort in cases.groupby("snapshot"):
        history = PredictionWindow.from_date(snapshot).select_history(orders)
        history = history[history.StoreID.isin(cohort.StoreID)]
        counts = history.groupby("StoreID").SalesPersonID.nunique()
        last = (
            history.sort_values(["OrderDate", "SalesOrderID"])
            .groupby("StoreID")
            .tail(1)
        )
        for row in last.itertuples(index=False):
            profiles.append(
                {
                    "StoreID": row.StoreID,
                    "snapshot": snapshot,
                    "salesperson": str(row.SalesPersonID),
                    "territory": str(row.TerritoryID),
                    "observed_assignment_change": int(counts.loc[row.StoreID] > 1),
                }
            )
    profiles = cases[KEYS + ["churn", "n_orders_3m"]].merge(
        pd.DataFrame(profiles), on=KEYS, validate="one_to_one"
    )
    require(len(profiles) == 1375, "Incomplete assignment population")
    matches = read_csv(source / "matches.csv")
    matches = matches[matches.k.eq(3)]
    pair = matches.merge(profiles, on=KEYS, validate="many_to_one")
    pair = pair.merge(
        profiles.rename(columns={"StoreID": "control_StoreID"}),
        on=["control_StoreID", "snapshot"],
        suffixes=("", "_control"),
        validate="many_to_one",
    )
    # Average controls once per target before summarizing; repeated matches are dependent.
    per_target = (
        pair.groupby(["comparison", *KEYS])
        .agg(
            target_change=("observed_assignment_change", "first"),
            control_change=("observed_assignment_change_control", "mean"),
        )
        .reset_index()
    )
    summary = per_target.groupby(["comparison", "snapshot"])[
        ["target_change", "control_change"]
    ].agg(["size", "mean"])
    category_rows = []
    for scope, subset in [
        ("overall", profiles),
        ("recent_purchase", profiles[profiles.n_orders_3m.gt(0)]),
    ]:
        for dimension in ["salesperson", "territory", "observed_assignment_change"]:
            for (snapshot, value), group in subset.groupby(["snapshot", dimension]):
                positives = int(group.churn.sum())
                category_rows.append(
                    {
                        "scope": scope,
                        "snapshot": snapshot,
                        "dimension": dimension,
                        "category": value,
                        "rows": len(group),
                        "churners": positives,
                        "resellers": group.StoreID.nunique(),
                        "churn_rate": positives / len(group),
                        "small_cell": len(group) < 30
                        or positives < 10
                        or len(group) - positives < 10,
                    }
                )
    shares = []
    for dimension in ["salesperson", "territory"]:
        for (comparison, snapshot), frame in pair.groupby(["comparison", "snapshot"]):
            for category in sorted(
                set(frame[dimension]) | set(frame[dimension + "_control"])
            ):
                target_share = (
                    frame.drop_duplicates(KEYS)[dimension].eq(category).mean()
                )
                control_share = frame[dimension + "_control"].eq(category).mean()
                shares.append(
                    {
                        "comparison": comparison,
                        "snapshot": snapshot,
                        "dimension": dimension,
                        "category": category,
                        "target_share": target_share,
                        "control_share": control_share,
                    }
                )
    out = new_output_directory(output)
    for name, table in {
        "profiles": profiles,
        "paired_assignments": pair,
        "per_target_changes": per_target,
        "category_rates": pd.DataFrame(category_rows),
        "matched_category_shares": pd.DataFrame(shares),
    }.items():
        table.to_csv(out / f"{name}.csv", index=False)
    summary.to_csv(out / "matched_changes.csv")
    write_manifest(
        out,
        {
            "stage": "historical_assignment_screen",
            "models_fitted": 0,
            "cutoffs_selected": 0,
            "october_outcomes_accessed": False,
            "inputs": input_identity(
                {
                    "history_manifest": source / "manifest.json",
                    "orders": orders_path,
                    "protocol": ROOT / "comparisons/persistent_error_investigation.md",
                }
            ),
            "implementation": input_identity({"script": Path(__file__)}),
        },
    )
    print(summary.to_string())
    print(
        profiles.groupby("snapshot")
        .observed_assignment_change.agg(["sum", "count"])
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

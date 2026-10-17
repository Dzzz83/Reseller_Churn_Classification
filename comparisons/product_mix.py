"""Describe historical SKU identities beyond category shares and diversity counts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, verify_baseline
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    write_manifest,
)
from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS
from reseller_churn.data.prediction_window import PredictionWindow

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def historical_product_shares(keys, orders, lines):
    require(not keys.duplicated(KEYS).any(), "Duplicate profile case")
    require(
        set(keys.snapshot.dt.strftime("%Y-%m-%d")).issubset(TRAINING_SNAPSHOTS),
        "Only approved development snapshots",
    )
    require(
        not orders.SalesOrderID.duplicated().any()
        and not lines.SalesOrderDetailID.duplicated().any(),
        "Duplicate source keys",
    )
    profiles = []
    for snapshot, cohort in keys.groupby("snapshot"):
        history = PredictionWindow.from_date(snapshot).select_history(orders)
        history = history[history.StoreID.isin(cohort.StoreID)]
        detail = lines[lines.SalesOrderID.isin(history.SalesOrderID)].merge(
            history[["StoreID", "SalesOrderID"]],
            on="SalesOrderID",
            validate="many_to_one",
        )
        totals = (
            detail.groupby("SalesOrderID").LineTotal.sum().reindex(history.SalesOrderID)
        )
        np.testing.assert_allclose(totals, history.SubTotal, rtol=1e-8, atol=0.02)
        latest = (
            history.sort_values(["OrderDate", "SalesOrderID"])
            .groupby("StoreID")
            .tail(1)
        )
        for scope, part in [
            ("annual", detail),
            ("latest_order", detail[detail.SalesOrderID.isin(latest.SalesOrderID)]),
        ]:
            revenue = (
                part.groupby(["StoreID", "ProductID"])
                .LineTotal.sum()
                .rename("line_revenue")
                .reset_index()
            )
            denominator = revenue.groupby("StoreID").line_revenue.transform("sum")
            require(denominator.gt(0).all(), "Invalid share denominator")
            revenue["share"] = revenue.line_revenue / denominator
            revenue["snapshot"], revenue["scope"] = snapshot, scope
            require(
                set(revenue.StoreID) == set(cohort.StoreID),
                "Missing eligible product history",
            )
            np.testing.assert_allclose(
                revenue.groupby("StoreID").share.sum(), 1.0, rtol=0, atol=1e-12
            )
            profiles.append(revenue)
    return pd.concat(profiles, ignore_index=True)


def matched_sku_tables(profiles, matches):
    summaries = []
    distances = []
    for scope, part in profiles.groupby("scope"):
        matrix = part.pivot(index=KEYS, columns="ProductID", values="share").fillna(0)
        products = matrix.columns.to_numpy()
        for (comparison, k), pairs in matches.groupby(["comparison", "k"]):
            targets = []
            controls = []
            snapshots = []
            control_presence = []
            for (store, snapshot), neighbors in pairs.groupby(KEYS):
                target = matrix.loc[(store, snapshot)].to_numpy()
                indices = pd.MultiIndex.from_frame(
                    neighbors[["control_StoreID", "snapshot"]].rename(
                        columns={"control_StoreID": "StoreID"}
                    )
                )
                control = matrix.reindex(indices).to_numpy()
                require(np.isfinite(control).all(), "Unmatched product profile")
                targets.append(target)
                controls.append(control.mean(axis=0))
                control_presence.append((control > 0).mean(axis=0))
                snapshots.append(snapshot)
                for neighbor, (_, row) in enumerate(neighbors.iterrows()):
                    distances.append(
                        {
                            "comparison": comparison,
                            "k": int(k),
                            "StoreID": store,
                            "snapshot": snapshot,
                            "control_StoreID": row.control_StoreID,
                            "scope": scope,
                            "baseline_distance": row.distance,
                            "product_total_variation": float(
                                np.abs(target - control[neighbor]).sum() / 2
                            ),
                        }
                    )
            targets, controls = np.array(targets), np.array(controls)
            control_presence = np.array(control_presence)
            for snapshot in [None, *sorted(set(snapshots))]:
                mask = (
                    np.ones(len(snapshots), dtype=bool)
                    if snapshot is None
                    else np.array([value == snapshot for value in snapshots])
                )
                target, control = targets[mask], controls[mask]
                difference = target - control
                for idx, product in enumerate(products):
                    summaries.append(
                        {
                            "comparison": comparison,
                            "k": int(k),
                            "scope": scope,
                            "snapshot": snapshot,
                            "aggregation": "overall"
                            if snapshot is None
                            else "snapshot",
                            "ProductID": int(product),
                            "target_cases": int(mask.sum()),
                            "target_mean_share": float(target[:, idx].mean()),
                            "control_mean_share": float(control[:, idx].mean()),
                            "mean_difference": float(difference[:, idx].mean()),
                            "median_difference": float(np.median(difference[:, idx])),
                            "target_presence_fraction": float(
                                (target[:, idx] > 0).mean()
                            ),
                            "control_mean_positive_fraction": float(
                                control_presence[mask, idx].mean()
                            ),
                            "one_point_positive_review_flag": bool(
                                difference[:, idx].mean() >= 0.01
                            ),
                            "one_point_negative_review_flag": bool(
                                difference[:, idx].mean() <= -0.01
                            ),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(distances)


def run(output):
    require(not output.exists(), "Preserve existing output")
    orders_path = (
        ROOT / "results/research-provenance-20261016/orders_clean_historical_bytes.csv"
    )
    features, _, audit = verify_baseline(
        ROOT / "results/repeated-grouped-precision", orders_path
    )
    preparation = json.loads(
        (ROOT / "results/development-precision-diagnosis/manifest.json").read_text()
    )
    lines_path = ROOT / "inputs/regeneration-20261016/order_lines.csv"
    require(
        file_sha256(lines_path) == preparation["input_hashes"]["lines"],
        "Changed order lines",
    )
    orders = pd.read_csv(orders_path, parse_dates=["OrderDate"])
    orders = orders[orders.OrderDate < features.snapshot.max()]
    lines = pd.read_csv(lines_path)
    lines = lines[lines.SalesOrderID.isin(orders.SalesOrderID)]
    profiles = historical_product_shares(features[KEYS], orders, lines)
    match_path = ROOT / "results/persistent-error-investigation-20261016/matches.csv"
    matches = read_csv(match_path)
    summary, distances = matched_sku_tables(profiles, matches)
    out = new_output_directory(output)
    profiles.to_csv(out / "case_sku_shares.csv", index=False)
    summary.to_csv(out / "matched_sku_summary.csv", index=False)
    distances.to_csv(out / "product_distances.csv", index=False)
    flags = (
        summary[(summary.aggregation == "snapshot") & (summary.k == 3)]
        .groupby(["comparison", "scope", "ProductID"])
        .agg(
            snapshots=("snapshot", "nunique"),
            positive_review_snapshots=("one_point_positive_review_flag", "sum"),
            negative_review_snapshots=("one_point_negative_review_flag", "sum"),
            minimum_targets=("target_cases", "min"),
        )
        .reset_index()
    )
    flags.to_csv(out / "all_product_review_flags.csv", index=False)
    write_manifest(
        out,
        {
            "stage": "product_identity_diagnosis",
            "models_fitted": 0,
            "cutoffs_selected": 0,
            "october_outcomes_accessed": False,
            "baseline_audit": audit,
            "inputs": input_identity(
                {
                    "orders": orders_path,
                    "lines": lines_path,
                    "matches": match_path,
                    "protocol": ROOT / "comparisons/persistent_error_investigation.md",
                }
            ),
            "implementation": input_identity({"script": Path(__file__)}),
        },
    )
    print(
        "Products",
        profiles.ProductID.nunique(),
        "case/scope profiles",
        profiles.groupby(KEYS + ["scope"]).ngroups,
    )
    print(
        distances[distances.k == 3]
        .groupby(["comparison", "scope"])
        .product_total_variation.agg(["min", "median", "max"])
        .to_string()
    )
    print("All-four-snapshot review flags (not a selected feature set):")
    print(
        flags[
            (flags.snapshots == 4)
            & (
                (flags.positive_review_snapshots == 4)
                | (flags.negative_review_snapshots == 4)
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

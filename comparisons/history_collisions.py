"""Audit exact observable-history collisions, without model fitting or cutoffs."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, verify_baseline
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]
LINE_FIELDS = ["ProductID", "OrderQty", "UnitPrice", "UnitPriceDiscount", "LineTotal"]


def signature(values):
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def history_signatures(keys, orders, lines):
    require(not keys.duplicated(KEYS).any(), "Duplicate case")
    details = {
        int(order): sorted(part[LINE_FIELDS].itertuples(index=False, name=None))
        for order, part in lines.groupby("SalesOrderID")
    }
    rows = []
    for snapshot, cases in keys.groupby("snapshot"):
        past = orders[orders.OrderDate.lt(snapshot)]
        for store in cases.StoreID:
            purchases = past[past.StoreID.eq(store)]
            for scope in ("annual_history", "all_observed_history"):
                window = (
                    purchases
                    if scope == "all_observed_history"
                    else purchases[
                        purchases.OrderDate.ge(snapshot - pd.DateOffset(months=12))
                    ]
                )
                require(len(window) > 0, "Missing eligible history")
                history = sorted(
                    (
                        (snapshot - order.OrderDate).days,
                        order.SubTotal,
                        details[int(order.SalesOrderID)],
                    )
                    for order in window.itertuples()
                )
                rows.append(
                    dict(
                        StoreID=store,
                        snapshot=snapshot,
                        representation=scope,
                        signature=signature(history),
                    )
                )
    return pd.DataFrame(rows)


def collision_tables(cases):
    cells = (
        cases.groupby(["representation", "snapshot", "signature"])
        .agg(
            cell_rows=("StoreID", "size"),
            cell_churners=("churn", "sum"),
            cell_resellers=("StoreID", "nunique"),
        )
        .reset_index()
    )
    cells["mixed_outcome"] = cells.cell_churners.gt(0) & cells.cell_churners.lt(
        cells.cell_rows
    )
    joined = cases.merge(
        cells, on=["representation", "snapshot", "signature"], validate="many_to_one"
    )
    summaries = []
    for (representation, snapshot), frame in cells.groupby(
        ["representation", "snapshot"]
    ):
        duplicate = frame.cell_rows.gt(1)
        summaries.append(
            dict(
                representation=representation,
                snapshot=snapshot,
                signatures=len(frame),
                duplicate_cells=int(duplicate.sum()),
                duplicate_rows=int(frame.loc[duplicate, "cell_rows"].sum()),
                mixed_cells=int(frame.mixed_outcome.sum()),
                mixed_rows=int(frame.loc[frame.mixed_outcome, "cell_rows"].sum()),
            )
        )
    return joined, pd.DataFrame(summaries)


def run(output):
    orders_path = (
        ROOT / "results/research-provenance-20261016/orders_clean_historical_bytes.csv"
    )
    data, _, _ = verify_baseline(
        ROOT / "results/repeated-grouped-precision", orders_path
    )
    orders = pd.read_csv(orders_path, parse_dates=["OrderDate"])
    orders = orders[orders.OrderDate.lt(data.snapshot.max())]
    lines_path = ROOT / "inputs/regeneration-20261016/order_lines.csv"
    expected = json.loads(
        (ROOT / "results/development-precision-diagnosis/manifest.json").read_text()
    )["input_hashes"]["lines"]
    require(file_sha256(lines_path) == expected, "Changed source lines")
    lines = pd.read_csv(lines_path)
    lines = lines[lines.SalesOrderID.isin(orders.SalesOrderID)]
    signatures = history_signatures(data[KEYS], orders, lines)
    base = data[KEYS].copy()
    base["representation"] = "baseline_ten"
    base["signature"] = [
        signature([None if pd.isna(v) else float(v) for v in row])
        for row in data[list(FEATURES)].itertuples(index=False, name=None)
    ]
    cases = pd.concat([base, signatures]).merge(
        data[KEYS + ["churn"]], on=KEYS, validate="many_to_one"
    )
    annotated, summary = collision_tables(cases)
    errors = read_csv(
        ROOT / "results/persistent-error-investigation-20261016/case_histories.csv"
    )
    annotated = annotated.merge(
        errors[KEYS + ["alerts", "n_orders_3m"]], on=KEYS, validate="many_to_one"
    )
    predictions = read_csv(
        ROOT / "results/cadence-stability-20261016/validation_predictions.csv"
    )
    predictions = predictions[predictions.strategy.eq("baseline")]
    fold_summaries, fold_cases = [], []
    for (repeat, fold), frame in predictions.groupby(["repeat", "outer_fold"]):
        joined = cases.merge(
            frame[KEYS + ["score", "cutoff"]], on=KEYS, validate="many_to_one"
        )
        marked, table = collision_tables(joined)
        marked["repeat"], marked["outer_fold"] = repeat, fold
        marked["alert"] = marked.score.ge(marked.cutoff)
        fold_cases.append(marked)
        table["repeat"], table["outer_fold"] = repeat, fold
        fold_summaries.append(table)
    out = new_output_directory(output)
    annotated.to_csv(out / "case_collisions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(fold_cases).to_csv(out / "outer_fold_cases.csv", index=False)
    pd.concat(fold_summaries).to_csv(out / "outer_fold_collisions.csv", index=False)
    write_manifest(
        out,
        dict(
            stage="exact_history_collisions",
            models_fitted=0,
            cutoffs_selected=0,
            october_outcomes_accessed=False,
            inputs=input_identity(
                dict(
                    orders=orders_path,
                    lines=lines_path,
                    protocol=ROOT / "comparisons/persistent_error_investigation.md",
                )
            ),
            implementation=input_identity(dict(script=Path(__file__))),
        ),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

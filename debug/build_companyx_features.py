"""Clean CompanyX order extracts and derive features strictly before snapshots."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.validation_settings import LABELED_SNAPSHOTS

OUTPUT = ROOT / "results/debugging/companyx_features"
SOURCE = OUTPUT / "source"
DATA = ROOT / "results/codex-luna-verification-20261010-1/preparation/labeled_development.csv"
FEATURE_GROUPS = {
    "basket_volume": [
        "avg_qty_3m", "avg_qty_12m", "avg_lines_3m",
        "qty_ratio_3m", "net_unit_price_3m",
    ],
    "discounts": [
        "weighted_discount_3m", "discount_change_3m", "discounted_line_share_12m",
    ],
    "product_diversity": [
        "unique_products_3m", "unique_products_12m", "new_product_share_3m",
    ],
}


def clean_extracts(headers, lines, reference):
    """Retain the established clean-order population; validate the new details."""
    if headers["SalesOrderID"].duplicated().any():
        raise ValueError("Duplicate order headers.")
    if lines["SalesOrderDetailID"].duplicated().any():
        raise ValueError("Duplicate order lines.")
    if reference["SalesOrderID"].duplicated().any():
        raise ValueError("Duplicate reference orders.")
    if not set(lines["SalesOrderID"]).issubset(set(headers["SalesOrderID"])):
        raise ValueError("Orphan order lines.")
    if set(reference["SalesOrderID"]) - set(headers["SalesOrderID"]):
        raise ValueError("Source omits established clean orders.")
    excluded = headers[~headers["SalesOrderID"].isin(reference["SalesOrderID"])].copy()
    clean_headers = headers[headers["SalesOrderID"].isin(reference["SalesOrderID"])].copy()
    clean_lines = lines[lines["SalesOrderID"].isin(clean_headers["SalesOrderID"])].copy()
    if set(clean_headers["SalesOrderID"]) != set(clean_lines["SalesOrderID"]):
        raise ValueError("Accepted order has no detail lines.")
    required = ["ProductID", "OrderQty", "UnitPrice", "UnitPriceDiscount", "LineTotal"]
    if clean_lines[required].isna().any().any():
        raise ValueError("Accepted orders contain missing detail values.")
    if not clean_lines[["OrderQty", "UnitPrice", "LineTotal"]].gt(0).all().all():
        raise ValueError("Accepted orders contain nonpositive detail values.")
    if not clean_lines["UnitPriceDiscount"].between(0, 1).all():
        raise ValueError("Discount outside the fractional range.")
    expected_line_total = (
        clean_lines["OrderQty"] * clean_lines["UnitPrice"]
        * (1 - clean_lines["UnitPriceDiscount"])
    )
    np.testing.assert_allclose(expected_line_total, clean_lines["LineTotal"], rtol=1e-8, atol=1e-5)

    # Aggregate lines before comparing with header revenue; never sum repeated headers.
    totals = clean_lines.groupby("SalesOrderID").agg(
        qty=("OrderQty", "sum"), n_lines=("SalesOrderDetailID", "size"),
        avg_disc=("UnitPriceDiscount", "mean"), detail_revenue=("LineTotal", "sum"),
    )
    compared = clean_headers.merge(totals, on="SalesOrderID", validate="one_to_one")
    compared = compared.merge(
        reference[["SalesOrderID", "StoreID", "OrderDate", "SubTotal", "qty", "n_lines", "avg_disc"]],
        on="SalesOrderID", suffixes=("", "_reference"), validate="one_to_one",
    )
    for column in ["StoreID", "OrderDate", "qty", "n_lines"]:
        np.testing.assert_array_equal(compared[column], compared[f"{column}_reference"])
    for column in ["SubTotal", "avg_disc"]:
        np.testing.assert_allclose(
            compared[column], compared[f"{column}_reference"], rtol=1e-10, atol=1e-6
        )
    np.testing.assert_allclose(compared["SubTotal"], compared["detail_revenue"], rtol=1e-8, atol=.02)
    return clean_headers, clean_lines, excluded


def period_totals(headers, lines):
    orders = headers.groupby("StoreID")["SalesOrderID"].nunique()
    quantity = lines.groupby("StoreID")["OrderQty"].sum()
    line_count = lines.groupby("StoreID")["SalesOrderDetailID"].size()
    revenue = lines.groupby("StoreID")["LineTotal"].sum()
    gross = lines["OrderQty"] * lines["UnitPrice"]
    weighted_discount = (gross * lines["UnitPriceDiscount"]).groupby(lines["StoreID"]).sum()
    gross_by_store = gross.groupby(lines["StoreID"]).sum()
    return {
        "orders": orders, "quantity": quantity, "lines": line_count,
        "avg_qty": quantity / orders, "avg_lines": line_count / orders,
        "net_unit_price": revenue / quantity,
        "weighted_discount": weighted_discount / gross_by_store,
        "discounted_share": lines["UnitPriceDiscount"].gt(0).groupby(lines["StoreID"]).mean(),
        "products": lines.groupby("StoreID")["ProductID"].nunique(),
    }


def build_features(keys, headers, lines):
    if set(keys["snapshot"].dt.strftime("%Y-%m-%d")) - set(LABELED_SNAPSHOTS):
        raise ValueError("Only approved development snapshots are permitted.")
    details = lines.merge(
        headers[["SalesOrderID", "StoreID", "OrderDate"]],
        on="SalesOrderID", validate="many_to_one",
    )
    frames = []
    for snapshot, cohort in keys.groupby("snapshot"):
        history_start = snapshot - pd.DateOffset(months=12)
        recent_start = snapshot - pd.DateOffset(months=3)
        previous_start = snapshot - pd.DateOffset(months=6)
        history_headers = headers[
            headers["OrderDate"].ge(history_start) & headers["OrderDate"].lt(snapshot)
        ]
        history_lines = details[
            details["OrderDate"].ge(history_start) & details["OrderDate"].lt(snapshot)
        ]
        total = period_totals(history_headers, history_lines)
        recent = period_totals(
            history_headers[history_headers["OrderDate"].ge(recent_start)],
            history_lines[history_lines["OrderDate"].ge(recent_start)],
        )
        previous = period_totals(
            history_headers[
                history_headers["OrderDate"].ge(previous_start)
                & history_headers["OrderDate"].lt(recent_start)
            ],
            history_lines[
                history_lines["OrderDate"].ge(previous_start)
                & history_lines["OrderDate"].lt(recent_start)
            ],
        )
        result = cohort[["StoreID", "snapshot"]].copy()
        for name, series in {
            "avg_qty_3m": recent["avg_qty"], "avg_qty_12m": total["avg_qty"],
            "avg_lines_3m": recent["avg_lines"], "net_unit_price_3m": recent["net_unit_price"],
            "weighted_discount_3m": recent["weighted_discount"],
            "discounted_line_share_12m": total["discounted_share"],
            "unique_products_3m": recent["products"], "unique_products_12m": total["products"],
        }.items():
            result[name] = result["StoreID"].map(series)
        for name in ["unique_products_3m", "unique_products_12m"]:
            result[name] = result[name].fillna(0).astype(int)
        current_qty = result["StoreID"].map(recent["quantity"]).fillna(0)
        previous_qty = result["StoreID"].map(previous["quantity"])
        result["qty_ratio_3m"] = current_qty / previous_qty.where(previous_qty.gt(0))
        result["discount_change_3m"] = (
            result["weighted_discount_3m"]
            - result["StoreID"].map(previous["weighted_discount"])
        )
        old_products = history_lines[history_lines["OrderDate"].lt(recent_start)].groupby("StoreID")["ProductID"].agg(set)
        current_products = history_lines[history_lines["OrderDate"].ge(recent_start)].groupby("StoreID")["ProductID"].agg(set)
        new_share = {}
        for store_id, products in current_products.items():
            old = old_products.get(store_id, set())
            new_share[store_id] = len(products - old) / len(products)
        result["new_product_share_3m"] = result["StoreID"].map(new_share)
        last_order = history_headers.sort_values(["OrderDate", "SalesOrderID"]).groupby("StoreID").tail(1).set_index("StoreID")
        result["last_salesperson"] = result["StoreID"].map(last_order["SalesPersonID"])
        result["last_territory"] = result["StoreID"].map(last_order["TerritoryID"])
        # Reconcile the source history against model inputs before making new features.
        result["check_n_orders_12m"] = result["StoreID"].map(total["orders"]).fillna(0)
        result["check_n_orders_3m"] = result["StoreID"].map(recent["orders"]).fillna(0)
        hist_revenue = history_headers.groupby("StoreID")["SubTotal"].sum()
        result["check_revenue_12m"] = result["StoreID"].map(hist_revenue).fillna(0)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Freeze the feature groups before reading outcome correlations or fitting models.
    write_json(OUTPUT / "feature_protocol.json", {
        "groups": FEATURE_GROUPS, "baseline": "seven-feature weighted logistic regression",
        "precision_floor": .30, "selection_grid": "0.00 to 1.00 in 0.01 steps",
        "groups_tested_separately": True, "tuning": False,
        "availability_assumption": "Past order headers and order-line values describe facts known when the order was placed; recording-time history is unavailable",
    })
    headers = pd.read_csv(SOURCE / "order_headers_raw.csv", parse_dates=["OrderDate", "DueDate", "ShipDate"])
    lines = pd.read_csv(SOURCE / "order_lines_raw.csv")
    if not headers["OrderDate"].lt("2013-04-01").all():
        raise ValueError("Extract reaches outside the experiment's historical cutoff.")
    reference = pd.read_csv(ROOT / "datasets/orders_clean.csv", parse_dates=["OrderDate"])
    reference = reference[reference["OrderDate"].lt("2013-04-01")]
    headers, lines, excluded = clean_extracts(headers, lines, reference)
    data = pd.read_csv(DATA, parse_dates=["snapshot"])
    features = build_features(data[["StoreID", "snapshot"]], headers, lines)
    features = features.merge(data, on=["StoreID", "snapshot"], validate="one_to_one")
    for name in ["n_orders_12m", "n_orders_3m", "revenue_12m"]:
        np.testing.assert_allclose(features[f"check_{name}"], features[name], rtol=1e-10, atol=1e-5)
        features = features.drop(columns=[f"check_{name}"])
    features.to_csv(OUTPUT / "development_features.csv", index=False)
    headers.to_csv(OUTPUT / "clean_order_headers.csv", index=False)
    lines.to_csv(OUTPUT / "clean_order_lines.csv", index=False)
    excluded.to_csv(OUTPUT / "excluded_orders.csv", index=False)
    write_json(OUTPUT / "cleaning_manifest.json", {
        "raw_orders": len(headers) + len(excluded), "clean_orders": len(headers),
        "clean_lines": len(lines), "excluded_order_ids": excluded["SalesOrderID"].astype(int).tolist(),
        "accepted_orders_current_status_6": int(headers["Status"].eq(6).sum()),
        "clean_days_to_ship_unique": sorted((headers["ShipDate"] - headers["OrderDate"]).dt.days.unique().tolist()),
        "development_rows": len(features), "protected_labels_accessed": False,
        "cleaning": "Established clean-order IDs preserved; details validated and sums reconciled; current status does not alter eligibility or labels",
        "checks": "Unique keys, no orphans, detail value ranges, line formula, reference counts/discount/revenue, historical model-input reconciliation",
        "input_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [SOURCE / "order_headers_raw.csv", SOURCE / "order_lines_raw.csv", DATA, ROOT / "datasets/orders_clean.csv"]
        },
    })
    print(f"Cleaned {len(headers)} orders and {len(lines)} lines; built {len(features)} development rows and 11 new features")
    print(features[[name for group in FEATURE_GROUPS.values() for name in group]].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()

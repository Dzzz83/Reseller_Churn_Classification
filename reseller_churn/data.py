"""Historical source validation, final features, and development-only labels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reseller_churn.config import FEATURES, TRAINING_SNAPSHOTS

KEYS = ["StoreID", "snapshot"]
CATEGORIES = ["rev_bikes", "rev_components", "rev_clothing", "rev_accessories"]


def date(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError("Dates must be non-missing, timezone-naive calendar dates.")
    return result


def require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def unique_ids(frame: pd.DataFrame, columns, name: str) -> None:
    require_columns(frame, columns, name)
    if frame[list(columns)].isna().any().any() or frame.duplicated(list(columns)).any():
        raise ValueError(f"{name} has missing or duplicate keys.")
    for column in columns:
        if column == "snapshot":
            frame[column].map(date)
            continue
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(f"{name}: {column} must contain positive integer IDs.")


def finite_columns(frame: pd.DataFrame, columns, name: str, *, positive=False) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (positive and (values <= 0).any()):
        raise ValueError(f"{name} contains missing, nonfinite, or nonpositive values.")


@dataclass
class Sources:
    orders: pd.DataFrame
    stores: pd.DataFrame
    headers: pd.DataFrame
    lines: pd.DataFrame
    coverage_from: pd.Timestamp
    coverage_through: pd.Timestamp
    excluded_order_ids: tuple[int, ...] = ()

    def require_coverage(self, start, end) -> None:
        if date(start) < self.coverage_from or date(end) > self.coverage_through:
            raise ValueError(
                "Declared source coverage does not contain the complete required window."
            )


def load_sources(
    paths: dict[str, Path], coverage_from, coverage_through, needed_through
) -> Sources:
    """Reconcile only pre-cutoff source rows; future transactions never enter features.

    Coverage is an explicit completeness declaration, not inferred from the last
    transaction. Source headers may include established rejected orders; clean
    order IDs define the accepted population, preserving the original contract.
    """
    start, through, cutoff = (
        date(coverage_from),
        date(coverage_through),
        date(needed_through),
    )
    if start >= through or cutoff > through:
        raise ValueError("Invalid or insufficient source coverage.")
    orders = pd.read_csv(paths["orders"], parse_dates=["OrderDate"])
    headers = pd.read_csv(paths["headers"], parse_dates=["OrderDate"])
    stores = pd.read_csv(paths["stores"])
    lines = pd.read_csv(paths["lines"])
    require_columns(
        orders,
        [
            "StoreID",
            "SalesOrderID",
            "OrderDate",
            "SubTotal",
            "qty",
            "n_lines",
            *CATEGORIES,
        ],
        "orders",
    )
    require_columns(
        headers, ["StoreID", "SalesOrderID", "OrderDate", "SubTotal"], "headers"
    )
    require_columns(
        lines,
        [
            "SalesOrderID",
            "SalesOrderDetailID",
            "ProductID",
            "OrderQty",
            "UnitPrice",
            "UnitPriceDiscount",
            "LineTotal",
        ],
        "lines",
    )
    if orders.OrderDate.isna().any() or headers.OrderDate.isna().any():
        raise ValueError("Source order dates must be present.")
    if not lines.SalesOrderID.isin(headers.SalesOrderID).all():
        raise ValueError("Orphan order lines lack a source header.")
    orders = orders[orders.OrderDate.lt(cutoff)].copy()
    headers = headers[headers.OrderDate.lt(cutoff)].copy()
    # Restrict details by historical order IDs before validating accepted history.
    lines = lines[lines.SalesOrderID.isin(headers.SalesOrderID)].copy()
    unique_ids(orders, ["SalesOrderID"], "orders")
    unique_ids(headers, ["SalesOrderID"], "headers")
    unique_ids(stores, ["StoreID"], "stores")
    unique_ids(lines, ["SalesOrderDetailID"], "lines")
    if not orders.SalesOrderID.isin(headers.SalesOrderID).all():
        raise ValueError("Headers omit accepted clean orders.")
    excluded = tuple(
        sorted(
            headers.loc[
                ~headers.SalesOrderID.isin(orders.SalesOrderID), "SalesOrderID"
            ].astype(int)
        )
    )
    headers = headers[headers.SalesOrderID.isin(orders.SalesOrderID)].copy()
    lines = lines[lines.SalesOrderID.isin(orders.SalesOrderID)].copy()
    for frame, name in [(orders, "orders"), (headers, "headers")]:
        finite_columns(frame, ["StoreID", "SubTotal"], name, positive=True)
        if not frame.StoreID.isin(stores.StoreID).all():
            raise ValueError(f"{name} references unknown stores.")
    finite_columns(orders, ["qty", "n_lines"], "orders", positive=True)
    finite_columns(orders, CATEGORIES, "orders")
    if (orders[CATEGORIES] < 0).any().any():
        raise ValueError("Category revenue cannot be negative.")
    if not np.allclose(
        orders[CATEGORIES].sum(axis=1), orders.SubTotal, atol=0.01, rtol=0
    ):
        raise ValueError("Category revenue does not reconcile with clean subtotal.")
    finite_columns(
        lines,
        ["ProductID", "OrderQty", "UnitPrice", "LineTotal"],
        "lines",
        positive=True,
    )
    finite_columns(lines, ["UnitPriceDiscount"], "lines")
    if not lines.UnitPriceDiscount.between(0, 1).all():
        raise ValueError("Discounts must be fractions in [0, 1].")
    expected = lines.OrderQty * lines.UnitPrice * (1 - lines.UnitPriceDiscount)
    if not np.allclose(expected, lines.LineTotal, atol=1e-5, rtol=1e-8):
        raise ValueError("Order-line revenue formula does not reconcile.")
    totals = lines.groupby("SalesOrderID").agg(
        qty=("OrderQty", "sum"),
        n_lines=("SalesOrderDetailID", "size"),
        detail_revenue=("LineTotal", "sum"),
    )
    reconciled = orders.merge(
        headers, on="SalesOrderID", suffixes=("", "_header"), validate="one_to_one"
    )
    reconciled = reconciled.merge(
        totals,
        on="SalesOrderID",
        how="left",
        suffixes=("", "_detail"),
        validate="one_to_one",
    )
    for name in ["StoreID", "OrderDate"]:
        if not reconciled[name].eq(reconciled[f"{name}_header"]).all():
            raise ValueError(f"Clean orders and headers disagree on {name}.")
    for left, right, atol in [
        ("SubTotal", "SubTotal_header", 1e-6),
        ("SubTotal", "detail_revenue", 0.02),
        ("qty", "qty_detail", 0),
        ("n_lines", "n_lines_detail", 0),
    ]:
        if not np.allclose(reconciled[left], reconciled[right], atol=atol, rtol=1e-8):
            raise ValueError(f"Source reconciliation failed for {left}/{right}.")
    return Sources(orders, stores, headers, lines, start, through, excluded)


def build_features(source: Sources, snapshot) -> pd.DataFrame:
    """Build only the ten frozen inputs, preserving their original formulas."""
    snapshot = date(snapshot)
    history_start = snapshot - pd.DateOffset(months=12)
    source.require_coverage(history_start, snapshot)
    history = source.orders[
        source.orders.OrderDate.ge(history_start) & source.orders.OrderDate.lt(snapshot)
    ]
    eligible = history.loc[
        history.OrderDate.ge(snapshot - pd.DateOffset(months=6)), "StoreID"
    ].unique()
    history = history[history.StoreID.isin(eligible)]
    features = (
        history.groupby("StoreID")
        .agg(
            last_order=("OrderDate", "max"),
            revenue_12m=("SubTotal", "sum"),
            rev_bikes=("rev_bikes", "sum"),
            rev_accessories=("rev_accessories", "sum"),
            rev_clothing=("rev_clothing", "sum"),
        )
        .reset_index()
    )
    features["snapshot"] = snapshot
    features["recency_days"] = (snapshot - features.last_order).dt.days
    recent_start = snapshot - pd.DateOffset(months=3)
    recent = (
        history[history.OrderDate.ge(recent_start)]
        .groupby("StoreID")
        .agg(
            n_orders_3m=("SalesOrderID", "nunique"),
            revenue_3m=("SubTotal", "sum"),
        )
    )
    for name in ["n_orders_3m", "revenue_3m"]:
        features[name] = features.StoreID.map(recent[name]).fillna(0)
    features["n_orders_3m"] = features.n_orders_3m.astype(int)
    for category in ["bikes", "accessories", "clothing"]:
        features[f"share_{category}"] = (
            features[f"rev_{category}"] / features.revenue_12m
        )

    details = source.lines.merge(
        source.headers[["SalesOrderID", "StoreID", "OrderDate"]],
        on="SalesOrderID",
        validate="many_to_one",
    )

    def period(start, end):
        rows = details[details.OrderDate.ge(start) & details.OrderDate.lt(end)]
        gross = rows.OrderQty * rows.UnitPrice
        discount = (gross * rows.UnitPriceDiscount).groupby(rows.StoreID).sum()
        denominator = gross.groupby(rows.StoreID).sum()
        share = rows.UnitPriceDiscount.gt(0).groupby(rows.StoreID).mean()
        return discount / denominator, share

    recent_discount, _ = period(recent_start, snapshot)
    previous_discount, _ = period(snapshot - pd.DateOffset(months=6), recent_start)
    _, annual_share = period(history_start, snapshot)
    features["weighted_discount_3m"] = features.StoreID.map(recent_discount)
    features["discount_change_3m"] = (
        features.weighted_discount_3m - features.StoreID.map(previous_discount)
    )
    features["discounted_line_share_12m"] = features.StoreID.map(annual_share)
    return features[[*KEYS, *FEATURES]].sort_values("StoreID").reset_index(drop=True)


def build_development(source: Sources) -> pd.DataFrame:
    """Construct only approved historical training outcomes, never final labels."""
    parts = []
    for snapshot in TRAINING_SNAPSHOTS:
        start = date(snapshot)
        end = start + pd.DateOffset(months=6)
        source.require_coverage(start, end)
        features = build_features(source, start)
        retained = source.orders.loc[
            source.orders.OrderDate.ge(start) & source.orders.OrderDate.lt(end),
            "StoreID",
        ].unique()
        features["churn"] = (~features.StoreID.isin(retained)).astype(int)
        parts.append(features)
    return pd.concat(parts, ignore_index=True)

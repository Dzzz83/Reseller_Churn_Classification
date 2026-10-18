"""Historical source validation, final features, and development-only labels."""

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reseller_churn.config.frozen_model import FEATURES, TRAINING_SNAPSHOTS
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.eligibility import get_eligible_store_ids
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.data.validation import (
    validate_category_revenue,
    validate_order_dates,
)
from reseller_churn.dates import DateLike
from reseller_churn.dates import parse_calendar_date as date
from reseller_churn.features.builders import (
    build_discount_features,
    build_purchase_features,
)

KEYS = ["StoreID", "snapshot"]
CATEGORIES = ["rev_bikes", "rev_components", "rev_clothing", "rev_accessories"]
ORDER_COLUMNS = [
    "StoreID",
    "SalesOrderID",
    "OrderDate",
    "SubTotal",
    "qty",
    "n_lines",
    *CATEGORIES,
]
HEADER_COLUMNS = ["StoreID", "SalesOrderID", "OrderDate", "SubTotal"]
LINE_COLUMNS = [
    "SalesOrderID",
    "SalesOrderDetailID",
    "ProductID",
    "OrderQty",
    "UnitPrice",
    "UnitPriceDiscount",
    "LineTotal",
]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def unique_ids(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    columns = list(columns)
    require_columns(frame, columns, name)
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
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


def finite_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
    *,
    positive: bool = False,
) -> None:
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
    loaded_through: pd.Timestamp
    excluded_order_ids: tuple[int, ...] = ()

    def require_coverage(self, start: DateLike, end: DateLike) -> None:
        if date(end) > self.loaded_through:
            raise ValueError("Required window extends beyond the loaded source cutoff.")
        if date(start) < self.coverage_from or date(end) > self.coverage_through:
            raise ValueError(
                "Declared source coverage does not contain the complete required window."
            )


def _read_source_files(
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, path in paths.items():
        with Path(path).open(newline="") as stream:
            columns = next(csv.reader(stream))
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate source columns: {name}")
        if {
            "churn",
            "actual_churn",
            "test_churn",
            "future_orders",
            "predicted_churn",
            "model score",
            "predicted churn",
        } & {c.lower() for c in columns}:
            raise ValueError("Source inputs must not contain outcomes or predictions.")
    orders = pd.read_csv(
        paths["orders"], parse_dates=["OrderDate"], float_precision="round_trip"
    )
    headers = pd.read_csv(
        paths["headers"], parse_dates=["OrderDate"], float_precision="round_trip"
    )
    stores = pd.read_csv(paths["stores"])
    lines = pd.read_csv(paths["lines"], float_precision="round_trip")
    require_columns(orders, ORDER_COLUMNS, "orders")
    require_columns(headers, HEADER_COLUMNS, "headers")
    require_columns(lines, LINE_COLUMNS, "lines")
    validate_order_dates(orders.OrderDate)
    validate_order_dates(headers.OrderDate)
    if not lines.SalesOrderID.isin(headers.SalesOrderID).all():
        raise ValueError("Orphan order lines lack a source header.")
    return orders, stores, headers, lines


def _historical_rows(
    orders: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return source rows belonging to orders strictly before the cutoff."""
    orders = orders[orders.OrderDate.lt(cutoff)].copy()
    headers = headers[headers.OrderDate.lt(cutoff)].copy()
    lines = lines[lines.SalesOrderID.isin(headers.SalesOrderID)].copy()
    return orders, headers, lines


def _accepted_order_population(
    orders: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
    stores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Restrict raw headers and lines to the established clean order IDs."""
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
    return headers, lines, excluded


def _validate_source_values(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
) -> None:
    for frame, name in [(orders, "orders"), (headers, "headers")]:
        finite_columns(frame, ["StoreID", "SubTotal"], name, positive=True)
        if not frame.StoreID.isin(stores.StoreID).all():
            raise ValueError(f"{name} references unknown stores.")
    finite_columns(orders, ["qty", "n_lines"], "orders", positive=True)
    validate_category_revenue(orders)
    finite_columns(
        lines,
        ["ProductID", "OrderQty", "UnitPrice", "LineTotal"],
        "lines",
        positive=True,
    )
    finite_columns(lines, ["UnitPriceDiscount"], "lines")
    if not lines.UnitPriceDiscount.between(0, 1).all():
        raise ValueError("Discounts must be fractions in [0, 1].")
    for frame, columns in [
        (orders, ["qty", "n_lines"]),
        (lines, ["SalesOrderID", "ProductID", "OrderQty"]),
    ]:
        values = frame[columns].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError("Source counts and IDs must be positive integers.")
    expected = lines.OrderQty * lines.UnitPrice * (1 - lines.UnitPriceDiscount)
    if not np.allclose(expected, lines.LineTotal, atol=1e-5, rtol=1e-8):
        raise ValueError("Order-line revenue formula does not reconcile.")


def _reconcile_accepted_orders(
    orders: pd.DataFrame, headers: pd.DataFrame, lines: pd.DataFrame
) -> None:
    """Verify that accepted order summaries agree with raw headers and lines."""
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


def load_sources(
    paths: Mapping[str, Path],
    coverage_from: DateLike,
    coverage_through: DateLike,
    needed_through: DateLike,
    *,
    approved_excluded_order_ids: Iterable[int] = (),
) -> Sources:
    """Load and reconcile pre-cutoff source rows without exposing future facts.

    Coverage is an explicit completeness declaration, not inferred from the last
    transaction. Source headers may include established rejected orders; clean
    order IDs define the accepted population, preserving the original contract.
    """
    start = date(coverage_from)
    through = date(coverage_through)
    cutoff = date(needed_through)
    if start >= through or cutoff > through or cutoff <= start:
        raise ValueError("Invalid or insufficient source coverage.")

    orders, stores, headers, lines = _read_source_files(paths)
    orders, headers, lines = _historical_rows(orders, headers, lines, cutoff)
    headers, lines, excluded = _accepted_order_population(
        orders, headers, lines, stores
    )
    approved = pd.DataFrame({"SalesOrderID": list(approved_excluded_order_ids)})
    unique_ids(approved, ["SalesOrderID"], "approved exclusions")
    unexpected = set(excluded) - set(approved.SalesOrderID)
    if unexpected:
        raise ValueError(
            f"Unexpected orders absent from clean input: {sorted(unexpected)}. "
            "Supply independently approved excluded order IDs."
        )
    _validate_source_values(orders, stores, headers, lines)
    _reconcile_accepted_orders(orders, headers, lines)
    return Sources(orders, stores, headers, lines, start, through, cutoff, excluded)


def build_features(source: Sources, snapshot: DateLike) -> pd.DataFrame:
    """Combine independent purchase and discount inputs for eligible resellers."""
    snapshot = date(snapshot)
    window = PredictionWindow.from_date(snapshot)
    source.require_coverage(window.history_start, snapshot)
    history = window.select_history(source.orders)
    eligible = get_eligible_store_ids(source.orders, window)
    history = history[history.StoreID.isin(eligible)]
    purchases = build_purchase_features(history, snapshot)

    details = source.lines.merge(
        source.headers[["SalesOrderID", "StoreID", "OrderDate"]],
        on="SalesOrderID",
        validate="many_to_one",
    )
    details = details[details.StoreID.isin(purchases.StoreID)]
    discounts = build_discount_features(details, snapshot)
    features = purchases.merge(
        discounts, on="StoreID", how="left", validate="one_to_one"
    )
    features["snapshot"] = snapshot
    return features[[*KEYS, *FEATURES]].sort_values("StoreID").reset_index(drop=True)


def build_development(source: Sources) -> pd.DataFrame:
    """Construct only approved historical training outcomes, never final labels."""
    parts = []
    for snapshot in TRAINING_SNAPSHOTS:
        window = PredictionWindow.from_date(snapshot)
        source.require_coverage(window.label_start, window.label_end)
        features = build_features(source, window.snapshot)
        labels = build_churn_labels(source.orders, window)
        parts.append(
            features.merge(
                labels[["StoreID", "churn"]],
                on="StoreID",
                validate="one_to_one",
            )
        )
    return pd.concat(parts, ignore_index=True)

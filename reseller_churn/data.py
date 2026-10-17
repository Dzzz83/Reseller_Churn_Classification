"""Historical source validation and development-only training labels."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reseller_churn.config import TRAINING_SNAPSHOTS
from reseller_churn.dates import DateLike, parse_calendar_date
from reseller_churn.features import build_features

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


def require_unique_keys(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Require nonmissing, unique keys with positive integer IDs and valid snapshots."""
    columns = list(columns)
    require_columns(frame, columns, name)
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f"{name} has missing or duplicate keys.")
    for column in columns:
        if column == "snapshot":
            frame[column].map(parse_calendar_date)
            continue
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(f"{name}: {column} must contain positive integer IDs.")


def require_finite_values(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
    *,
    positive: bool = False,
) -> None:
    """Reject missing/infinite numbers and, when requested, values at or below zero."""
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (positive and (values <= 0).any()):
        raise ValueError(f"{name} contains missing, nonfinite, or nonpositive values.")


@dataclass
class Sources:
    """Reconciled accepted orders and lines, with declared complete source coverage."""

    orders: pd.DataFrame
    stores: pd.DataFrame
    headers: pd.DataFrame
    lines: pd.DataFrame
    coverage_from: pd.Timestamp
    coverage_through: pd.Timestamp
    excluded_order_ids: tuple[int, ...] = ()

    def require_coverage(
        self, window_start: DateLike, window_end_exclusive: DateLike
    ) -> None:
        """Require complete source coverage of [window_start, window_end_exclusive)."""
        if (
            parse_calendar_date(window_start) < self.coverage_from
            or parse_calendar_date(window_end_exclusive) > self.coverage_through
        ):
            raise ValueError(
                "Declared source coverage does not contain the complete required window."
            )


def _read_source_files(
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(paths["orders"], parse_dates=["OrderDate"])
    headers = pd.read_csv(paths["headers"], parse_dates=["OrderDate"])
    stores = pd.read_csv(paths["stores"])
    lines = pd.read_csv(paths["lines"])
    require_columns(orders, ORDER_COLUMNS, "orders")
    require_columns(headers, HEADER_COLUMNS, "headers")
    require_columns(lines, LINE_COLUMNS, "lines")
    if orders.OrderDate.isna().any() or headers.OrderDate.isna().any():
        raise ValueError("Source order dates must be present.")
    if not lines.SalesOrderID.isin(headers.SalesOrderID).all():
        raise ValueError("Orphan order lines lack a source header.")
    return orders, stores, headers, lines


def _historical_rows(
    orders: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep orders before end_date and their matching lines."""
    orders = orders[orders.OrderDate.lt(end_date)].copy()
    headers = headers[headers.OrderDate.lt(end_date)].copy()
    lines = lines[lines.SalesOrderID.isin(headers.SalesOrderID)].copy()
    return orders, headers, lines


def _accepted_order_population(
    orders: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
    stores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Restrict raw headers and lines to the established clean order IDs."""
    require_unique_keys(orders, ["SalesOrderID"], "orders")
    require_unique_keys(headers, ["SalesOrderID"], "headers")
    require_unique_keys(stores, ["StoreID"], "stores")
    require_unique_keys(lines, ["SalesOrderDetailID"], "lines")
    if not orders.SalesOrderID.isin(headers.SalesOrderID).all():
        raise ValueError("Headers omit accepted clean orders.")
    excluded_order_ids = tuple(
        sorted(
            headers.loc[
                ~headers.SalesOrderID.isin(orders.SalesOrderID), "SalesOrderID"
            ].astype(int)
        )
    )
    headers = headers[headers.SalesOrderID.isin(orders.SalesOrderID)].copy()
    lines = lines[lines.SalesOrderID.isin(orders.SalesOrderID)].copy()
    return headers, lines, excluded_order_ids


def _validate_source_values(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    headers: pd.DataFrame,
    lines: pd.DataFrame,
) -> None:
    for frame, name in [(orders, "orders"), (headers, "headers")]:
        require_finite_values(frame, ["StoreID", "SubTotal"], name, positive=True)
        if not frame.StoreID.isin(stores.StoreID).all():
            raise ValueError(f"{name} references unknown stores.")
    require_finite_values(orders, ["qty", "n_lines"], "orders", positive=True)
    require_finite_values(orders, CATEGORIES, "orders")
    if (orders[CATEGORIES] < 0).any().any():
        raise ValueError("Category revenue cannot be negative.")
    if not np.allclose(
        orders[CATEGORIES].sum(axis=1), orders.SubTotal, atol=0.01, rtol=0
    ):
        raise ValueError("Category revenue does not reconcile with clean subtotal.")
    require_finite_values(
        lines,
        ["ProductID", "OrderQty", "UnitPrice", "LineTotal"],
        "lines",
        positive=True,
    )
    require_finite_values(lines, ["UnitPriceDiscount"], "lines")
    if not lines.UnitPriceDiscount.between(0, 1).all():
        raise ValueError("Discounts must be fractions in [0, 1].")
    expected_line_revenue = (
        lines.OrderQty * lines.UnitPrice * (1 - lines.UnitPriceDiscount)
    )
    if not np.allclose(expected_line_revenue, lines.LineTotal, atol=1e-5, rtol=1e-8):
        raise ValueError("Order-line revenue formula does not reconcile.")


def _reconcile_accepted_orders(
    orders: pd.DataFrame, headers: pd.DataFrame, lines: pd.DataFrame
) -> None:
    """Verify that accepted order summaries agree with raw headers and lines."""
    line_totals_by_order = lines.groupby("SalesOrderID").agg(
        qty=("OrderQty", "sum"),
        n_lines=("SalesOrderDetailID", "size"),
        detail_revenue=("LineTotal", "sum"),
    )
    reconciled = orders.merge(
        headers, on="SalesOrderID", suffixes=("", "_header"), validate="one_to_one"
    )
    reconciled = reconciled.merge(
        line_totals_by_order,
        on="SalesOrderID",
        how="left",
        suffixes=("", "_detail"),
        validate="one_to_one",
    )
    for name in ["StoreID", "OrderDate"]:
        if not reconciled[name].eq(reconciled[f"{name}_header"]).all():
            raise ValueError(f"Clean orders and headers disagree on {name}.")
    for order_column, source_column, absolute_tolerance in [
        ("SubTotal", "SubTotal_header", 1e-6),
        ("SubTotal", "detail_revenue", 0.02),
        ("qty", "qty_detail", 0),
        ("n_lines", "n_lines_detail", 0),
    ]:
        if not np.allclose(
            reconciled[order_column],
            reconciled[source_column],
            atol=absolute_tolerance,
            rtol=1e-8,
        ):
            raise ValueError(
                f"Source reconciliation failed for {order_column}/{source_column}."
            )


def load_sources(
    paths: Mapping[str, Path],
    coverage_from: DateLike,
    coverage_through: DateLike,
    end_date: DateLike,
) -> Sources:
    """Load and reconcile source rows before the required exclusive endpoint.

    Coverage is an explicit completeness declaration, not inferred from the last
    transaction. Source headers may include established rejected orders; clean
    order IDs define the accepted population, preserving the original contract.
    """
    coverage_start = parse_calendar_date(coverage_from)
    coverage_end_exclusive = parse_calendar_date(coverage_through)
    # Orders on or after end_date are excluded.
    end_date = parse_calendar_date(end_date)
    if coverage_start >= coverage_end_exclusive or end_date > coverage_end_exclusive:
        raise ValueError("Invalid or insufficient source coverage.")

    orders, stores, headers, lines = _read_source_files(paths)
    orders, headers, lines = _historical_rows(orders, headers, lines, end_date)
    headers, lines, excluded_order_ids = _accepted_order_population(
        orders, headers, lines, stores
    )
    _validate_source_values(orders, stores, headers, lines)
    _reconcile_accepted_orders(orders, headers, lines)
    return Sources(
        orders=orders,
        stores=stores,
        headers=headers,
        lines=lines,
        coverage_from=coverage_start,
        coverage_through=coverage_end_exclusive,
        excluded_order_ids=excluded_order_ids,
    )


def build_development(sources: Sources) -> pd.DataFrame:
    """Pair historical features with six-month labels for approved snapshots only."""
    training_samples_by_snapshot = []
    for snapshot in TRAINING_SNAPSHOTS:
        snapshot_date = parse_calendar_date(snapshot)
        outcome_end_exclusive = snapshot_date + pd.DateOffset(months=6)
        sources.require_coverage(snapshot_date, outcome_end_exclusive)
        training_samples = build_features(sources, snapshot_date)
        stores_with_future_purchases = sources.orders.loc[
            sources.orders.OrderDate.ge(snapshot_date)
            & sources.orders.OrderDate.lt(outcome_end_exclusive),
            "StoreID",
        ].unique()
        training_samples["churn"] = (
            ~training_samples.StoreID.isin(stores_with_future_purchases)
        ).astype(int)
        training_samples_by_snapshot.append(training_samples)
    return pd.concat(training_samples_by_snapshot, ignore_index=True)

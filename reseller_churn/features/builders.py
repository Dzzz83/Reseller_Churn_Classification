"""Independent purchase and discount calculations for the frozen feature contract."""

import pandas as pd

from reseller_churn.config.frozen_model import DISCOUNT_FEATURES, PURCHASE_FEATURES


def build_purchase_features(
    history: pd.DataFrame, snapshot: pd.Timestamp
) -> pd.DataFrame:
    """Return StoreID and seven features from eligible stores' twelve-month history."""
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
    features["recency_days"] = (snapshot - features.last_order).dt.days
    start_date = snapshot - pd.DateOffset(months=3)
    recent_orders = (
        history[history.OrderDate.ge(start_date)]
        .groupby("StoreID")
        .agg(
            n_orders_3m=("SalesOrderID", "nunique"),
            revenue_3m=("SubTotal", "sum"),
        )
    )
    for name in ["n_orders_3m", "revenue_3m"]:
        features[name] = features.StoreID.map(recent_orders[name]).fillna(0)
    features["n_orders_3m"] = features.n_orders_3m.astype(int)
    for category in ["bikes", "accessories", "clothing"]:
        features[f"share_{category}"] = (
            features[f"rev_{category}"] / features.revenue_12m
        )
    return features[["StoreID", *PURCHASE_FEATURES]]


def _weighted_discount(
    lines: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    """Weight each line's discount by quantity times unit price before discount."""
    # Orders on or after end_date are excluded.
    lines = lines[lines.OrderDate.ge(start_date) & lines.OrderDate.lt(end_date)]
    gross_values = lines.OrderQty * lines.UnitPrice
    discount_totals = (
        (gross_values * lines.UnitPriceDiscount).groupby(lines.StoreID).sum()
    )
    gross_totals = gross_values.groupby(lines.StoreID).sum()
    return discount_totals / gross_totals


def build_discount_features(
    lines: pd.DataFrame, snapshot: pd.Timestamp
) -> pd.DataFrame:
    """Return StoreID and three features from lines with store IDs and order dates.

    Missing purchase periods remain missing; they are not zero-discount purchases.
    """
    start_date = snapshot - pd.DateOffset(months=12)
    history = lines[lines.OrderDate.ge(start_date) & lines.OrderDate.lt(snapshot)]
    recent_start = snapshot - pd.DateOffset(months=3)
    previous_start = snapshot - pd.DateOffset(months=6)
    recent_discounts = _weighted_discount(history, recent_start, snapshot)
    previous_discounts = _weighted_discount(history, previous_start, recent_start)

    # Fraction of order lines with a discount in the last 12 months.
    line_shares = history.UnitPriceDiscount.gt(0).groupby(history.StoreID).mean()
    features = line_shares.rename("discount_frequency_12m").to_frame()
    # The series are indexed by StoreID, so assignment aligns each store's values.
    features["weighted_discount_3m"] = recent_discounts
    features["discount_change_3m"] = recent_discounts - previous_discounts
    return features.reset_index()[["StoreID", *DISCOUNT_FEATURES]]

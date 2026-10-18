"""Value checks shared by strict inputs and historical research preparation."""

import numpy as np
import pandas as pd

CATEGORY_COLUMNS = ("rev_bikes", "rev_components", "rev_clothing", "rev_accessories")


def validate_order_dates(values: pd.Series) -> None:
    """Preserve the trained feature formulas by accepting calendar dates only."""
    dates = pd.to_datetime(values, errors="raise")
    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError("Order dates must be present, timezone-naive and at midnight.")


def validate_category_revenue(orders: pd.DataFrame) -> None:
    values = orders[list(CATEGORY_COLUMNS)].to_numpy(dtype=float)
    subtotal = orders.SubTotal.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid category revenue: values must be finite and nonnegative.")
    if not np.isfinite(subtotal).all() or (subtotal <= 0).any():
        raise ValueError("Order subtotals must be finite and positive.")
    if not np.allclose(values.sum(axis=1), subtotal, atol=0.01, rtol=0):
        raise ValueError("Category revenue does not reconcile with clean subtotal.")

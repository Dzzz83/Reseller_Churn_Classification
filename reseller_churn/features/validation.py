"""Validation of the fixed ten historical model inputs."""

import numpy as np
import pandas as pd

from reseller_churn.config.frozen_model import DISCOUNT_FEATURES, FEATURES
from reseller_churn.data.preparation import KEYS, date, require_columns, unique_ids


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, [*KEYS, *FEATURES], "features")
    result = frame.copy()
    unique_ids(result, KEYS, "features")
    if result.empty:
        raise ValueError("Feature population is empty.")
    result["snapshot"] = result.snapshot.map(date)
    unique_ids(result, KEYS, "features")
    for name in FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise")
        values = result[name].to_numpy(dtype=float)
        if np.isinf(values).any() or (
            name not in DISCOUNT_FEATURES and np.isnan(values).any()
        ):
            raise ValueError(f"Invalid feature values: {name}")
    for name in ["n_orders_3m", "recency_days"]:
        values = result[name].to_numpy()
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid counts/days: {name}")
    for name in ["revenue_3m", "revenue_12m"]:
        if (result[name] < 0).any():
            raise ValueError(f"Negative revenue: {name}")
    for name in [
        "share_bikes",
        "share_accessories",
        "share_clothing",
        "weighted_discount_3m",
        "discount_frequency_12m",
    ]:
        if not result[name].dropna().between(0, 1).all():
            raise ValueError(f"Feature outside [0, 1]: {name}")
    if not result.discount_change_3m.dropna().between(-1, 1).all():
        raise ValueError("Discount change outside [-1, 1].")
    eligibility_days = (
        result.snapshot - result.snapshot.map(lambda t: t - pd.DateOffset(months=6))
    ).dt.days
    if not (
        (result.recency_days >= 1) & (result.recency_days <= eligibility_days)
    ).all():
        raise ValueError("Recency does not describe an eligible historical account.")
    if (result.revenue_12m <= 0).any() or (
        result.revenue_3m > result.revenue_12m + 0.02
    ).any():
        raise ValueError("Recent revenue must not exceed positive annual revenue.")
    shares = result[["share_bikes", "share_accessories", "share_clothing"]].sum(axis=1)
    if (shares > 1 + 1e-6).any():
        raise ValueError("Category shares sum to more than one.")
    no_recent_orders = result.n_orders_3m.eq(0)
    if (result.loc[no_recent_orders, "revenue_3m"].abs() > 0.02).any() or (
        result.loc[~no_recent_orders, "revenue_3m"] <= 0
    ).any():
        raise ValueError("Recent order counts and revenue disagree.")
    return result

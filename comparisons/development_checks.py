"""Boundaries for approved development research, separate from final evaluation."""

import numpy as np
import pandas as pd

from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS

KEYS = ["StoreID", "snapshot"]
FORBIDDEN_PREDICTORS = {*KEYS, "churn", "actual_churn", "test_churn", "future_orders"}


def validate_development_cases(frame, predictors=()):
    """Validate a full development table or subset without altering its rows.

    Matrix checks cannot establish source completeness or historical recording
    availability. Those still require source/formula review and boundary tests.
    """
    required = {*KEYS, "churn", *predictors}
    if required - set(frame) or frame.empty:
        raise ValueError("Missing development cases or columns")
    if {"actual_churn", "test_churn", "future_orders"} & set(frame):
        raise ValueError("Test/outcome-derived columns are not development inputs")
    if set(predictors) & FORBIDDEN_PREDICTORS:
        raise ValueError("Identifiers and outcomes cannot be predictors")
    dates = pd.to_datetime(frame.snapshot, errors="raise")
    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
        or not dates.dt.strftime("%Y-%m-%d").isin(TRAINING_SNAPSHOTS).all()
    ):
        raise ValueError("Only approved development snapshots are permitted")
    keys = frame[KEYS].assign(snapshot=dates)
    stores = pd.to_numeric(frame.StoreID, errors="raise").to_numpy(dtype=float)
    if (
        keys.isna().any().any()
        or keys.duplicated().any()
        or not (np.isfinite(stores) & (stores > 0) & (stores == np.floor(stores))).all()
    ):
        raise ValueError(
            "Development cases require unique valid reseller/snapshot keys"
        )
    if not frame.churn.isin([0, 1]).all():
        raise ValueError("Development labels must be complete and binary")
    if predictors and np.isinf(frame[list(predictors)].to_numpy(dtype=float)).any():
        raise ValueError("Infinite features are invalid; native missingness is allowed")


def validate_development_split(train, valid, *, design="grouped"):
    """Keep groups separate; optionally enforce matured-label forecasting too."""
    validate_development_cases(train)
    validate_development_cases(valid)
    if set(train.StoreID) & set(valid.StoreID):
        raise ValueError("Reseller leakage across development split")
    if design not in {"grouped", "temporal_grouped"}:
        raise ValueError("Declare grouped or temporal_grouped validation design")
    if train.churn.nunique() != 2:
        raise ValueError("Training subset requires both outcome classes")
    if design == "temporal_grouped":
        matured = pd.to_datetime(train.snapshot) + pd.DateOffset(months=6)
        if matured.max() > pd.to_datetime(valid.snapshot).min():
            raise ValueError("Training labels are not mature by validation prediction")


def validate_feature_times(times, snapshot):
    """Check explicitly supplied source event times against the prediction boundary."""
    dates = pd.to_datetime(times, errors="raise")
    boundary = pd.Timestamp(snapshot)
    if dates.isna().any() or dates.dt.tz is not None or boundary.tzinfo is not None:
        raise ValueError("Feature event times must be present and timezone-naive")
    if not dates.lt(boundary).all():
        raise ValueError("Feature events must precede prediction; end is exclusive")

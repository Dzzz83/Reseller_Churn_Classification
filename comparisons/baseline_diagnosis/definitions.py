"""Predeclared populations, policies, and descriptive profile definitions."""

from pathlib import Path

from reseller_churn.config.frozen_model import FEATURES, TRAINING_SNAPSHOTS

ROOT = Path(__file__).resolve().parents[2]

KEYS = ["StoreID", "snapshot"]

REPEATS = (42, 78, 88)

POLICIES = {"original_baseline": "inner_selected", "original_fixed_006": "fixed_006"}

SLICES = {
    "activity": ("recent_purchase", "no_recent_purchase"),
    "annual_orders": ("1", "2", "3+"),
    "history_span": ("0-90 days", "91-180 days", "181+ days"),
    **{f"{name}_availability": ("available", "missing") for name in FEATURES[-3:]},
    "snapshot_slice": TRAINING_SNAPSHOTS,
}

RANK_BANDS = ("top 10%", "10-25%", "25-50%", "50-75%", "bottom 25%")

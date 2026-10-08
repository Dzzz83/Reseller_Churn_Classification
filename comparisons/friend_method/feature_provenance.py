"""Feature provenance is separated from whether a feature predicts churn.

The CSV contains a single store profile without effective-from dates. Such
fields may be valid historical descriptors, but this cannot be established
from the source file alone. Keep them available as a diagnostic variant
rather than silently accepting them as leakage-safe.
"""
from dataclasses import dataclass

import pandas as pd

from comparisons.friend_method.corrected_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


UNVERIFIED_STORE_NUMERIC = (
    "AnnualSales",
    "AnnualRevenue",
    "SquareFeet",
    "NumberEmployees",
)
UNVERIFIED_STORE_CATEGORICAL = (
    "BusinessType",
    "Specialty",
    "Brands",
    "Internet",
    "store_salesperson",
)


@dataclass(frozen=True)
class FeatureScope:
    name: str
    numeric_candidates: tuple[str, ...]
    categorical_features: tuple[str, ...]


def attach_feature_scope(
    frame: pd.DataFrame,
    stores: pd.DataFrame,
    include_unverified_profile: bool,
) -> tuple[pd.DataFrame, FeatureScope]:
    if not include_unverified_profile:
        return frame.copy(), FeatureScope(
            name="historical_only",
            numeric_candidates=tuple(
                name for name in NUMERIC_FEATURES
                if name in frame.columns
            ),
            categorical_features=CATEGORICAL_FEATURES,
        )

    store_cols = (
        "StoreID",
        "SalesPersonID",
        *UNVERIFIED_STORE_NUMERIC,
        *UNVERIFIED_STORE_CATEGORICAL[:-1],
    )
    missing = set(store_cols) - set(stores.columns)
    if missing:
        raise ValueError(
            "Store profile missing required columns: "
            + ", ".join(sorted(missing))
        )

    profile = stores[list(store_cols)].rename(
        columns={"SalesPersonID": "store_salesperson"}
    )
    result = frame.merge(
        profile,
        on="StoreID",
        how="left",
        validate="many_to_one",
    )
    if result[list(UNVERIFIED_STORE_NUMERIC)].isna().any().any():
        raise AssertionError("Missing store profile values")

    return result, FeatureScope(
        name="unverified_store_profile_diagnostic",
        numeric_candidates=tuple(
            name for name in (
                *NUMERIC_FEATURES,
                *UNVERIFIED_STORE_NUMERIC,
            )
            if name in result.columns
        ),
        categorical_features=(
            "TerritoryID",
            *UNVERIFIED_STORE_CATEGORICAL,
        ),
    )

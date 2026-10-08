"""Corrected, historical-only reconstruction of the friend's feature variants.

The original notebooks remain unchanged in pnn-re1506/CurrentTopicTest.
This implementation intentionally excludes all nonhistorical store-profile
attributes. Only stable YearOpened is read from stores; order territory and
salesperson come from the last order *before* each prediction snapshot.
"""
from dataclasses import dataclass

import pandas as pd

from reseller_churn.data.churn_labels import ChurnLabelBuilder
from reseller_churn.data.prediction_window import PredictionWindow


CATEGORY_COLUMNS = {
    "share_bikes": "rev_bikes",
    "share_components": "rev_components",
    "share_clothing": "rev_clothing",
    "share_accessories": "rev_accessories",
}

CATEGORICAL_FEATURES = ("TerritoryID", "last_order_salesperson")

NUMERIC_FEATURES = (
    "recency_days",
    "tenure_days",
    "n_orders_total",
    "revenue_total",
    "aov",
    "avg_qty",
    "avg_lines",
    "avg_disc",
    "n_orders_3m",
    "n_orders_6m",
    "n_orders_12m",
    "revenue_3m",
    "revenue_6m",
    "revenue_12m",
    "rev_trend",
    "mean_gap",
    "std_gap",
    "overdue_ratio",
    "share_bikes",
    "share_components",
    "share_clothing",
    "share_accessories",
    "store_age",
)


@dataclass(frozen=True)
class HistoricalSnapshotBuilder:
    """Rebuild the two observation-window variants without future features."""

    orders: pd.DataFrame
    stores: pd.DataFrame

    def build(
        self,
        snapshots: tuple[str, ...],
        feature_window: str,
    ) -> pd.DataFrame:
        if feature_window not in ("full", "obs6"):
            raise ValueError("Expected 'full' or 'obs6'")

        frames = [
            self._build_one(pd.Timestamp(t), feature_window)
            for t in snapshots
        ]
        result = pd.concat(frames, ignore_index=True)

        if result.duplicated(["snapshot", "StoreID"]).any():
            raise AssertionError("Duplicate reseller/snapshot observations")

        return result

    def _build_one(
        self,
        snapshot: pd.Timestamp,
        feature_window: str,
    ) -> pd.DataFrame:
        window = PredictionWindow.from_date(snapshot)
        eligible = ChurnLabelBuilder().build(self.orders, window)

        history = self.orders.loc[
            self.orders["OrderDate"] < snapshot
        ].copy()
        if feature_window == "obs6":
            history = history.loc[
                history["OrderDate"]
                >= snapshot - pd.DateOffset(months=6)
            ].copy()

        history = history[
            history["StoreID"].isin(eligible["StoreID"])
        ].sort_values(
            ["StoreID", "OrderDate", "SalesOrderID"]
        )

        if history.empty:
            raise ValueError(f"No historical orders for {snapshot}")

        grouped = history.groupby("StoreID", sort=True)

        features = grouped.agg(
            first_order=("OrderDate", "min"),
            last_order=("OrderDate", "max"),
            n_orders_total=("SalesOrderID", "nunique"),
            revenue_total=("SubTotal", "sum"),
            aov=("SubTotal", "mean"),
            avg_qty=("qty", "mean"),
            avg_lines=("n_lines", "mean"),
            avg_disc=("avg_disc", "mean"),
            TerritoryID=("TerritoryID", "last"),
            last_order_salesperson=("SalesPersonID", "last"),
        ).reset_index()

        features["recency_days"] = (
            snapshot - features["last_order"]
        ).dt.days
        features["tenure_days"] = (
            snapshot - features["first_order"]
        ).dt.days

        half = 6 if feature_window == "full" else 3

        for months in (half, half * 2):
            window_orders = history[
                history["OrderDate"]
                >= snapshot - pd.DateOffset(months=months)
            ]
            recent = window_orders.groupby("StoreID").agg(
                **{
                    f"n_orders_{months}m": (
                        "SalesOrderID", "nunique"
                    ),
                    f"revenue_{months}m": (
                        "SubTotal", "sum"
                    ),
                }
            ).reset_index()
            features = features.merge(
                recent, on="StoreID", how="left", validate="one_to_one"
            )
            features[
                [f"n_orders_{months}m", f"revenue_{months}m"]
            ] = features[
                [f"n_orders_{months}m", f"revenue_{months}m"]
            ].fillna(0)

        current = features[f"revenue_{half}m"]
        previous = (
            features[f"revenue_{half * 2}m"] - current
        )
        # Division is intentionally undefined without earlier revenue.
        features["rev_trend"] = current.div(
            previous.where(previous > 0)
        )

        distinct_order_dates = (
            history[["StoreID", "OrderDate"]]
            .drop_duplicates()
            .sort_values(["StoreID", "OrderDate"])
        )
        distinct_order_dates["gap"] = (
            distinct_order_dates.groupby("StoreID")[
                "OrderDate"
            ].diff().dt.days
        )
        gap_features = distinct_order_dates.groupby(
            "StoreID"
        ).agg(
            mean_gap=("gap", "mean"),
            std_gap=("gap", "std"),
        ).reset_index()
        features = features.merge(
            gap_features,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )
        features["overdue_ratio"] = (
            features["recency_days"] / features["mean_gap"]
        )

        category_revenue = grouped[
            list(CATEGORY_COLUMNS.values())
        ].sum().reset_index()
        features = features.merge(
            category_revenue,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )
        for name, source in CATEGORY_COLUMNS.items():
            features[name] = (
                features[source] / features["revenue_total"]
            )

        features = features.merge(
            self.stores[["StoreID", "YearOpened"]],
            on="StoreID",
            how="left",
            validate="one_to_one",
        )
        if features["YearOpened"].isna().any():
            raise AssertionError("Missing YearOpened for an eligible reseller")
        features["store_age"] = (
            snapshot.year - features["YearOpened"]
        )

        features = features.merge(
            eligible[["StoreID", "churn"]],
            on="StoreID",
            validate="one_to_one",
        )

        features.insert(1, "snapshot", snapshot.strftime("%Y-%m-%d"))

        # The variants have different recent-window features.
        available_numeric = [
            feature for feature in NUMERIC_FEATURES
            if feature in features.columns
        ]
        return features[
            ["StoreID", "snapshot", "churn"]
            + available_numeric
            + list(CATEGORICAL_FEATURES)
        ].copy()

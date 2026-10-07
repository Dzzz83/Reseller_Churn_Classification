import numpy as np
import pandas as pd

from reseller_churn.step_01_data.prediction_window import PredictionWindow
from reseller_churn.step_01_data.eligibility import ResellerEligibility


CATEGORY_REVENUE_COLUMNS = {
    "bikes": "rev_bikes",
    "components": "rev_components",
    "clothing": "rev_clothing",
    "accessories": "rev_accessories",
}


class FeaturePipeline:
    """Build all reseller features from orders available before each snapshot.

    The public build method reads like the feature-engineering pipeline:
    base RFM -> recent activity -> purchase gaps -> revenue trend ->
    product shares -> store age.
    """

    def build(
        self,
        orders: pd.DataFrame,
        stores: pd.DataFrame,
        snapshots: tuple[str, ...] | list[str],
    ) -> pd.DataFrame:
        snapshot_features = [
            self._build_snapshot(
                orders=orders,
                stores=stores,
                snapshot=snapshot,
            )
            for snapshot in snapshots
        ]

        result = pd.concat(
            snapshot_features,
            ignore_index=True,
        )

        if result.duplicated(["StoreID", "snapshot"]).any():
            raise AssertionError(
                "Feature pipeline produced duplicate StoreID/snapshot rows."
            )

        return result

    def _build_snapshot(
        self,
        orders: pd.DataFrame,
        stores: pd.DataFrame,
        snapshot: str,
    ) -> pd.DataFrame:
        window = PredictionWindow.from_date(snapshot)
        history = window.select_history(orders)

        eligible_store_ids = ResellerEligibility.eligible_store_ids(
            orders,
            window,
        )

        history = history[
            history["StoreID"].isin(eligible_store_ids)
        ].copy()

        features = self._build_rfm_features(
            history,
            window,
        )

        features = self._add_recent_activity_features(
            features,
            history,
            window,
        )

        features = self._add_purchase_gap_features(
            features,
            history,
        )

        features = self._add_revenue_trend_features(
            features
        )

        features = self._add_product_share_features(
            features,
            history,
        )

        features = self._add_store_age_feature(
            features,
            stores,
            window,
        )

        return features[
            [
                "StoreID",
                "snapshot",
                "recency_days",
                "n_orders_12m",
                "revenue_12m",
                "n_orders_6m",
                "revenue_6m",
                "n_orders_3m",
                "revenue_3m",
                "mean_gap",
                "std_gap",
                "overdue_ratio",
                "has_previous_6m_revenue",
                "revenue_trend",
                "share_bikes",
                "share_components",
                "share_clothing",
                "share_accessories",
                "store_age",
            ]
        ]

    @staticmethod
    def _build_rfm_features(
        history: pd.DataFrame,
        window: PredictionWindow,
    ) -> pd.DataFrame:
        features = (
            history.groupby("StoreID")
            .agg(
                last_order=("OrderDate", "max"),
                n_orders_12m=("SalesOrderID", "nunique"),
                revenue_12m=("SubTotal", "sum"),
            )
            .reset_index()
        )

        features["recency_days"] = (
            window.snapshot - features["last_order"]
        ).dt.days

        features["snapshot"] = window.snapshot

        return features[
            [
                "StoreID",
                "snapshot",
                "last_order",
                "recency_days",
                "n_orders_12m",
                "revenue_12m",
            ]
        ]

    @staticmethod
    def _add_recent_activity_features(
        features: pd.DataFrame,
        history: pd.DataFrame,
        window: PredictionWindow,
    ) -> pd.DataFrame:
        result = features.copy()

        for months in (6, 3):
            period_start = (
                window.snapshot
                - pd.DateOffset(months=months)
            )

            recent_orders = history[
                history["OrderDate"] >= period_start
            ]

            recent_features = (
                recent_orders.groupby("StoreID")
                .agg(
                    **{
                        f"n_orders_{months}m": (
                            "SalesOrderID",
                            "nunique",
                        ),
                        f"revenue_{months}m": (
                            "SubTotal",
                            "sum",
                        ),
                    }
                )
                .reset_index()
            )

            result = result.merge(
                recent_features,
                on="StoreID",
                how="left",
                validate="one_to_one",
            )

            count_column = f"n_orders_{months}m"
            revenue_column = f"revenue_{months}m"

            result[
                [count_column, revenue_column]
            ] = result[
                [count_column, revenue_column]
            ].fillna(0)

            result[count_column] = (
                result[count_column].astype(int)
            )

        return result

    @staticmethod
    def _add_purchase_gap_features(
        features: pd.DataFrame,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        order_dates = (
            history[
                ["StoreID", "OrderDate"]
            ]
            .drop_duplicates()
            .sort_values(
                ["StoreID", "OrderDate"]
            )
            .copy()
        )

        order_dates["gap_days"] = (
            order_dates.groupby("StoreID")[
                "OrderDate"
            ]
            .diff()
            .dt.days
        )

        gap_features = (
            order_dates.groupby("StoreID")
            .agg(
                mean_gap=(
                    "gap_days",
                    "mean",
                ),
                std_gap=(
                    "gap_days",
                    "std",
                ),
            )
            .reset_index()
        )

        result = features.merge(
            gap_features,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        result["overdue_ratio"] = (
            result["recency_days"]
            / result["mean_gap"]
        )

        return result

    @staticmethod
    def _add_revenue_trend_features(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        result = features.copy()

        previous_revenue = (
            result["revenue_12m"]
            - result["revenue_6m"]
        )

        if (previous_revenue < -0.01).any():
            raise AssertionError(
                "Previous six-month revenue became negative."
            )

        previous_revenue = previous_revenue.clip(
            lower=0
        )

        has_previous_revenue = (
            previous_revenue > 0.01
        )

        result[
            "has_previous_6m_revenue"
        ] = has_previous_revenue.astype(int)

        result["revenue_trend"] = np.where(
            has_previous_revenue,
            (
                np.log1p(result["revenue_6m"])
                - np.log1p(previous_revenue)
            ),
            np.nan,
        )

        return result

    @staticmethod
    def _add_product_share_features(
        features: pd.DataFrame,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        revenue_columns = list(
            CATEGORY_REVENUE_COLUMNS.values()
        )

        category_revenue = (
            history.groupby("StoreID")[
                revenue_columns
            ]
            .sum()
            .reset_index()
        )

        result = features.merge(
            category_revenue,
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        for (
            category,
            revenue_column,
        ) in CATEGORY_REVENUE_COLUMNS.items():
            result[
                f"share_{category}"
            ] = (
                result[revenue_column]
                / result["revenue_12m"]
            )

        return result.drop(
            columns=revenue_columns
        )

    @staticmethod
    def _add_store_age_feature(
        features: pd.DataFrame,
        stores: pd.DataFrame,
        window: PredictionWindow,
    ) -> pd.DataFrame:
        result = features.merge(
            stores[
                ["StoreID", "YearOpened"]
            ],
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        if result["YearOpened"].isna().any():
            raise AssertionError(
                "A reseller feature row could not be matched to stores data."
            )

        result["store_age"] = (
            window.snapshot.year
            - result["YearOpened"]
        )

        return result.drop(
            columns=[
                "last_order",
                "YearOpened",
            ]
        )

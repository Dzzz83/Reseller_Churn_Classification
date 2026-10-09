import pandas as pd

from reseller_churn.data.eligibility import get_eligible_store_ids
from reseller_churn.data.prediction_window import PredictionWindow


class ChurnLabelBuilder:
    """Build the six-month inactivity target for eligible resellers."""

    def build(
        self,
        orders: pd.DataFrame,
        window: PredictionWindow,
    ) -> pd.DataFrame:
        eligible_store_ids = get_eligible_store_ids(
            orders,
            window,
        )

        future_orders = window.select_label_period(orders)

        future_order_counts = (
            future_orders.groupby("StoreID")["SalesOrderID"].nunique()
        )

        labels = pd.DataFrame({"StoreID": eligible_store_ids})

        labels["future_orders"] = (
            labels["StoreID"]
            .map(future_order_counts)
            .fillna(0)
            .astype(int)
        )

        labels["churn"] = (
            labels["future_orders"] == 0
        ).astype(int)

        return labels

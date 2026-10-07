import pandas as pd

from reseller_churn.step_02_data.prediction_window import PredictionWindow


class ResellerEligibility:
    """Identify resellers active in the six months before a snapshot."""

    @staticmethod
    def eligible_store_ids(
        orders: pd.DataFrame,
        window: PredictionWindow,
    ) -> list[int]:
        recent_orders = window.select_eligibility_period(orders)

        return sorted(
            recent_orders["StoreID"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

import pandas as pd

from reseller_churn.data.prediction_window import PredictionWindow


def get_eligible_store_ids(
    orders: pd.DataFrame,
    window: PredictionWindow,
) -> list[int]:
    """Return resellers active during the eligibility period."""

    recent_orders = window.select_eligibility_period(orders)

    return sorted(
        recent_orders["StoreID"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

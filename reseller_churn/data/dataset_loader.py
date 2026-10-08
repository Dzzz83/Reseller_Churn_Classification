import pandas as pd

from reseller_churn.config.project_paths import (
    LABELED_SNAPSHOTS_PATH,
    ORDERS_PATH,
    STORES_PATH,
)


class DatasetLoader:
    """Centralized CSV loading with consistent date parsing."""

    @staticmethod
    def load_orders() -> pd.DataFrame:
        return pd.read_csv(
            ORDERS_PATH,
            parse_dates=["OrderDate"],
            encoding="utf-8-sig",
        )

    @staticmethod
    def load_stores() -> pd.DataFrame:
        return pd.read_csv(STORES_PATH)

    @staticmethod
    def load_labeled_snapshots() -> pd.DataFrame:
        data = pd.read_csv(LABELED_SNAPSHOTS_PATH)
        data["snapshot"] = pd.to_datetime(
            data["snapshot"]
        ).dt.strftime("%Y-%m-%d")
        return data

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PredictionWindow:
    """Leakage-safe time boundaries around one prediction snapshot."""

    snapshot: pd.Timestamp
    history_months: int = 12
    eligibility_months: int = 6
    label_months: int = 6

    @classmethod
    def from_date(cls, snapshot: str | pd.Timestamp) -> "PredictionWindow":
        return cls(snapshot=pd.Timestamp(snapshot))

    @property
    def history_start(self) -> pd.Timestamp:
        return self.snapshot - pd.DateOffset(months=self.history_months)

    @property
    def eligibility_start(self) -> pd.Timestamp:
        return self.snapshot - pd.DateOffset(months=self.eligibility_months)

    @property
    def label_start(self) -> pd.Timestamp:
        return self.snapshot

    @property
    def label_end(self) -> pd.Timestamp:
        return self.snapshot + pd.DateOffset(months=self.label_months)

    def select_history(self, orders: pd.DataFrame) -> pd.DataFrame:
        return orders[
            (orders["OrderDate"] >= self.history_start)
            & (orders["OrderDate"] < self.snapshot)
        ].copy()

    def select_eligibility_period(self, orders: pd.DataFrame) -> pd.DataFrame:
        return orders[
            (orders["OrderDate"] >= self.eligibility_start)
            & (orders["OrderDate"] < self.snapshot)
        ].copy()

    def select_label_period(self, orders: pd.DataFrame) -> pd.DataFrame:
        return orders[
            (orders["OrderDate"] >= self.label_start)
            & (orders["OrderDate"] < self.label_end)
        ].copy()

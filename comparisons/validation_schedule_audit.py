"""Audit whether both validation strategies can be selected before one shared future holdout.

This checks the BEST-CASE calendar dates for a complete history window, not
model quality. If even the earliest possible dates do not fit, no later
snapshot schedule can fit either.
"""

from dataclasses import dataclass

import pandas as pd

from reseller_churn.config.validation_settings import FINAL_TEST_SNAPSHOT
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.churn_labels import build_churn_labels
from reseller_churn.data.prediction_window import PredictionWindow


@dataclass(frozen=True)
class ScheduleBounds:
    earliest_training: pd.Timestamp
    earliest_temporal_validation: pd.Timestamp
    temporal_results_available: pd.Timestamp
    latest_development_holdout: pd.Timestamp

    @property
    def has_independent_holdout(self) -> bool:
        return self.temporal_results_available <= self.latest_development_holdout


def assess_schedule(
    first_observed_date: str | pd.Timestamp,
    protected_test_date: str | pd.Timestamp,
    history_months: int = 12,
    label_months: int = 6,
) -> ScheduleBounds:
    """Find optimistic scheduling bounds, requiring complete history coverage.

    Temporal validation needs known training labels, then known validation
    labels, before either procedure can be compared on a shared later date.
    That shared date's labels must end by the protected test snapshot.
    """
    if history_months <= 0 or label_months <= 0:
        raise ValueError("History and label windows must be positive")

    first_training = (
        pd.Timestamp(first_observed_date)
        + pd.DateOffset(months=history_months)
    )
    first_temporal_validation = (
        first_training + pd.DateOffset(months=label_months)
    )

    return ScheduleBounds(
        earliest_training=first_training,
        earliest_temporal_validation=first_temporal_validation,
        temporal_results_available=(
            first_temporal_validation + pd.DateOffset(months=label_months)
        ),
        latest_development_holdout=(
            pd.Timestamp(protected_test_date)
            - pd.DateOffset(months=label_months)
        ),
    )



PARTIAL_HISTORY_CANDIDATE = (
    "2012-04-01",  # First training snapshot, about 10 months observed
    "2012-10-01",  # Temporal validation
    "2013-04-01",  # Shared future development holdout
)


def inspect_partial_history_candidate(
    orders: pd.DataFrame,
) -> list[tuple[str, int, int]]:
    """Check a relaxed-coverage design without using protected final-test labels."""
    first_training, validation, holdout = (
        pd.Timestamp(date) for date in PARTIAL_HISTORY_CANDIDATE
    )
    if (
        PredictionWindow.from_date(first_training).label_end > validation
        or PredictionWindow.from_date(validation).label_end > holdout
        or PredictionWindow.from_date(holdout).label_end
        > pd.Timestamp(FINAL_TEST_SNAPSHOT)
    ):
        raise AssertionError("Relaxed candidate violates label availability")

    rows = []
    for snapshot in PARTIAL_HISTORY_CANDIDATE:
        labels = build_churn_labels(orders, PredictionWindow.from_date(snapshot))
        rows.append((snapshot, len(labels), int(labels["churn"].sum())))
    return rows

def main() -> None:
    orders = DatasetLoader.load_orders()
    first_order = orders["OrderDate"].min().normalize()
    bounds = assess_schedule(first_order, FINAL_TEST_SNAPSHOT)

    print("=== 12-Month Validation Strategy Schedule Audit ===")
    print(f"First observed order:          {first_order.date()}")
    print(f"Earliest full-history train:   {bounds.earliest_training.date()}")
    print(
        "Earliest temporal validation: "
        f"{bounds.earliest_temporal_validation.date()}"
    )
    print(
        "Temporal results first known: "
        f"{bounds.temporal_results_available.date()}"
    )
    print(
        "Latest development holdout:  "
        f"{bounds.latest_development_holdout.date()}"
    )
    print(
        "Shared future holdout feasible: "
        f"{'YES' if bounds.has_independent_holdout else 'NO'}"
    )
    print(
        "Assumptions: complete 12-month history, six-month churn labels, "
        "temporal model selection completed before shared evaluation, "
        "no development label crossing the protected final-test date."
    )
    print()
    print("=== Partial-History Candidate (Different Assumption) ===")
    print("12-month lookback formulas remain, but early history is incomplete.")
    for snapshot, eligible, churners in inspect_partial_history_candidate(orders):
        print(f"{snapshot}: {eligible} eligible, {churners} churners")
    print("Feasibility only; not a model-performance comparison.")
    print("No models trained; no protected final-test labels accessed.")


if __name__ == "__main__":
    main()

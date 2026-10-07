import numpy as np
import pandas as pd

from reseller_churn.step_04_evaluation.metrics import (
    calculate_threshold_metrics,
)


class ThresholdSelector:
    """Compare classification thresholds without changing ranking metrics."""

    def __init__(
        self,
        thresholds: np.ndarray | None = None,
    ) -> None:
        self.thresholds = (
            thresholds
            if thresholds is not None
            else np.arange(0.01, 1.00, 0.01)
        )

    def evaluate(
        self,
        target,
        probabilities,
    ) -> pd.DataFrame:
        rows = []

        for threshold in self.thresholds:
            rows.append(
                {
                    "threshold":
                        float(threshold),
                    **calculate_threshold_metrics(
                        target,
                        probabilities,
                        float(threshold),
                    ),
                }
            )

        return pd.DataFrame(rows)

    def select_best_f1(
        self,
        target,
        probabilities,
    ) -> pd.Series:
        results = self.evaluate(
            target,
            probabilities,
        )

        return (
            results.sort_values(
                [
                    "f1",
                    "recall",
                    "precision",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

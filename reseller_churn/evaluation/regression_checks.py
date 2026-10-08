import numpy as np
import pandas as pd


class RegressionChecks:
    """Guard against accidental methodological changes during refactoring."""

    @staticmethod
    def assert_feature_frames_match(
        expected: pd.DataFrame,
        actual: pd.DataFrame,
    ) -> None:
        keys = ["StoreID", "snapshot"]

        expected = (
            expected.sort_values(keys)
            .reset_index(drop=True)
        )

        actual = (
            actual.sort_values(keys)
            .reset_index(drop=True)
        )

        if list(expected.columns) != list(actual.columns):
            raise AssertionError(
                "Refactored feature schema differs from the reference schema."
            )

        if len(expected) != len(actual):
            raise AssertionError(
                "Refactored feature row count differs from the reference."
            )

        for column in expected.columns:
            if column == "snapshot":
                left = pd.to_datetime(expected[column])
                right = pd.to_datetime(actual[column])
                if not left.equals(right):
                    raise AssertionError(
                        "Snapshot values changed during refactoring."
                    )
                continue

            if pd.api.types.is_numeric_dtype(expected[column]):
                left = expected[column].to_numpy(dtype=float)
                right = actual[column].to_numpy(dtype=float)

                if not np.allclose(
                    left,
                    right,
                    rtol=1e-10,
                    atol=1e-8,
                    equal_nan=True,
                ):
                    raise AssertionError(
                        f"Feature values changed for {column}."
                    )
            else:
                if not expected[column].equals(actual[column]):
                    raise AssertionError(
                        f"Values changed for {column}."
                    )

    @staticmethod
    def assert_score_matches(
        actual: float,
        expected: float,
        label: str,
        tolerance: float = 1e-10,
    ) -> None:
        if not np.isclose(
            actual,
            expected,
            rtol=0,
            atol=tolerance,
        ):
            raise AssertionError(
                f"{label}: expected {expected:.10f}, got {actual:.10f}."
            )

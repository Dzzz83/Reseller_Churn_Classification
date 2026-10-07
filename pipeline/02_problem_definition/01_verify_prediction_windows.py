from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_00_config.validation_settings import (
    FINAL_TEST_SNAPSHOT,
    LABELED_SNAPSHOTS,
)
from reseller_churn.step_01_data.prediction_window import PredictionWindow


def main() -> None:
    print("=== 02.1 Prediction Windows ===")

    final_test_snapshot = pd.Timestamp(
        FINAL_TEST_SNAPSHOT
    )

    for snapshot in LABELED_SNAPSHOTS:
        window = PredictionWindow.from_date(snapshot)

        assert window.label_start == window.snapshot
        assert window.label_end <= final_test_snapshot

        print()
        print(f"Snapshot:    {window.snapshot.date()}")
        print(
            f"History:     {window.history_start.date()} "
            f"to < {window.snapshot.date()}"
        )
        print(
            f"Eligibility: {window.eligibility_start.date()} "
            f"to < {window.snapshot.date()}"
        )
        print(
            f"Label:       {window.label_start.date()} "
            f"to < {window.label_end.date()}"
        )

    print()
    print(
        "[PASS] Every development label is fully resolved "
        "before or at the final-test snapshot."
    )
    print("Final-test labels were NOT accessed.")


if __name__ == "__main__":
    main()

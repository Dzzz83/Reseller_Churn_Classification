from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reseller_churn.config.validation_settings import (
    EXPECTED_LABELED_COUNTS,
)
from reseller_churn.data.churn_labels import ChurnLabelBuilder
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.prediction_window import PredictionWindow


def main() -> None:
    print("=== 02.2 Churn Label Verification ===")

    orders = DatasetLoader.load_orders()
    builder = ChurnLabelBuilder()

    for snapshot, (
        expected_rows,
        expected_churners,
    ) in EXPECTED_LABELED_COUNTS.items():
        labels = builder.build(
            orders=orders,
            window=PredictionWindow.from_date(snapshot),
        )

        actual = (
            len(labels),
            int(labels["churn"].sum()),
        )

        assert actual == (
            expected_rows,
            expected_churners,
        )

        print(
            f"[PASS] {snapshot}: "
            f"{actual[0]} eligible, "
            f"{actual[1]} churners"
        )

    print("Final-test labels were NOT accessed.")


if __name__ == "__main__":
    main()

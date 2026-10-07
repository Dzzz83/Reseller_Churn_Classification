
import pandas as pd

from define_windows import WINDOWS
from check_churn_labels import build_labels


DATA_PATH = "datasets/orders_clean.csv"


def report_group(name, group):
    total = len(group)
    churners = int(group["churn"].sum())
    rate = churners / total * 100 if total else 0

    print(
        f"{name:<28} "
        f"{total:>5} resellers | "
        f"{churners:>3} churners | "
        f"{rate:>6.2f}%"
    )


def main():
    orders = pd.read_csv(
        DATA_PATH,
        parse_dates=["OrderDate"],
    )

    first_orders = orders.groupby("StoreID")["OrderDate"].min()

    labels = {
        name: build_labels(orders, window)
        for name, window in WINDOWS.items()
    }

    print("=== Stage 1D.2: Population Investigation ===")

    transitions = [
        ("train", "validation"),
        ("validation", "test"),
    ]

    for previous, current in transitions:
        previous_labels = labels[previous]
        current_labels = labels[current].copy()

        previous_ids = set(previous_labels["StoreID"])
        current_ids = set(current_labels["StoreID"])

        retained_ids = previous_ids & current_ids
        new_ids = current_ids - previous_ids

        retained = current_labels[
            current_labels["StoreID"].isin(retained_ids)
        ]

        newcomers = current_labels[
            current_labels["StoreID"].isin(new_ids)
        ].copy()

        print(f"\n{'=' * 65}")
        print(f"{previous.upper()} -> {current.upper()}")
        print("=" * 65)

        report_group("All current resellers", current_labels)
        report_group("Retained from previous", retained)
        report_group("Not previously eligible", newcomers)

        # Separate first-observed from reactivated resellers.
        eligibility_start = (
            pd.Timestamp(WINDOWS[current]["observation_end"])
            - pd.DateOffset(months=6)
        )

        newcomers["first_observed"] = (
            newcomers["StoreID"].map(first_orders)
        )

        first_time = newcomers[
            newcomers["first_observed"] >= eligibility_start
        ]

        reactivated = newcomers[
            newcomers["first_observed"] < eligibility_start
        ]

        print("\nNewly eligible breakdown:")
        report_group("First observed in last 6m", first_time)
        report_group("Previously observed", reactivated)

        # Previous non-churners must be eligible in the next period.
        previous_non_churners = set(
            previous_labels.loc[
                previous_labels["churn"] == 0, "StoreID"
            ]
        )

        assert retained_ids == previous_non_churners
        assert retained_ids.isdisjoint(new_ids)
        assert retained_ids | new_ids == current_ids
        assert len(first_time) + len(reactivated) == len(newcomers)

        print("\n[PASS] Population transition checks")


if __name__ == "__main__":
    main()

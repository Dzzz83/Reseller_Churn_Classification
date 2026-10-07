from pathlib import Path

import pandas as pd


DATA_DIR = Path("datasets")
ORDERS_FILE = DATA_DIR / "orders_clean.csv"


WINDOWS = {
    "train": {
        "observation_start": "2011-07-01",
        "observation_end":   "2012-07-01",
        "label_start":       "2012-07-01",
        "label_end":         "2013-01-01",
    },
    "validation": {
        "observation_start": "2012-01-01",
        "observation_end":   "2013-01-01",
        "label_start":       "2013-01-01",
        "label_end":         "2013-07-01",
    },
    "test": {
        "observation_start": "2012-07-01",
        "observation_end":   "2013-07-01",
        "label_start":       "2013-07-01",
        "label_end":         "2014-01-01",
    },
}


def select_period(df, start, end):
    """Return rows in the half-open interval [start, end)."""
    return df[
        (df["OrderDate"] >= start)
        & (df["OrderDate"] < end)
    ]


def main():
    orders = pd.read_csv(
        ORDERS_FILE,
        parse_dates=["OrderDate"]
    )

    print("\n=== Stage 1B: Time Window Verification ===\n")

    all_passed = True

    for split_name, window in WINDOWS.items():
        obs_start = pd.Timestamp(window["observation_start"])
        obs_end = pd.Timestamp(window["observation_end"])
        label_start = pd.Timestamp(window["label_start"])
        label_end = pd.Timestamp(window["label_end"])

        observation = select_period(
            orders,
            obs_start,
            obs_end
        )

        label = select_period(
            orders,
            label_start,
            label_end
        )

        # --------------------------------------------
        # Basic checks
        # --------------------------------------------
        observation_months = (
            (obs_end.year - obs_start.year) * 12
            + obs_end.month
            - obs_start.month
        )

        label_months = (
            (label_end.year - label_start.year) * 12
            + label_end.month
            - label_start.month
        )

        correct_obs_length = observation_months == 12
        correct_label_length = label_months == 6
        periods_touch = obs_end == label_start

        split_passed = (
            correct_obs_length
            and correct_label_length
            and periods_touch
        )

        all_passed &= split_passed

        # --------------------------------------------
        # Report
        # --------------------------------------------
        print(f"{split_name.upper()}")
        print("-" * 45)

        print(
            f"Observation: {obs_start.date()} "
            f"-> {(obs_end - pd.Timedelta(days=1)).date()}"
        )

        print(
            f"Label:       {label_start.date()} "
            f"-> {(label_end - pd.Timedelta(days=1)).date()}"
        )

        print()

        print(
            f"Observation orders:    {len(observation):,}"
        )
        print(
            f"Observation resellers: {observation['StoreID'].nunique():,}"
        )

        print(
            f"Label-period orders:   {len(label):,}"
        )
        print(
            f"Label-period resellers:{label['StoreID'].nunique():,}"
        )

        print()

        print(
            f"[{'PASS' if correct_obs_length else 'FAIL'}] "
            f"Observation window = 12 months"
        )

        print(
            f"[{'PASS' if correct_label_length else 'FAIL'}] "
            f"Label window = 6 months"
        )

        print(
            f"[{'PASS' if periods_touch else 'FAIL'}] "
            f"Label starts immediately after observation"
        )

        print("\n")

    # ------------------------------------------------
    # Dataset coverage
    # ------------------------------------------------
    latest_required_date = max(
        pd.Timestamp(w["label_end"])
        for w in WINDOWS.values()
    )

    data_end = orders["OrderDate"].max()

    # label_end is exclusive, so latest actual required
    # date is the day before it.
    enough_data = data_end >= latest_required_date - pd.Timedelta(days=1)

    print("DATA COVERAGE")
    print("-" * 45)
    print(f"Latest order in dataset: {data_end.date()}")
    print(
        "Latest required label date: "
        f"{(latest_required_date - pd.Timedelta(days=1)).date()}"
    )

    print(
        f"[{'PASS' if enough_data else 'FAIL'}] "
        "Dataset covers all three windows"
    )

    all_passed &= enough_data

    print("\n=== Stage 1B Result ===")

    if all_passed:
        print("PASS: time-window design is valid.")
    else:
        print("FAIL: review the time-window definitions.")


if __name__ == "__main__":
    main()

import pandas as pd

from check_churn_labels import build_labels


DATA_PATH = "datasets/orders_clean.csv"

SNAPSHOTS = pd.date_range(
    "2012-07-01",
    "2013-10-01",
    freq="3MS",
)

HISTORY_MONTHS = 12
LABEL_MONTHS = 6

# Potential new test snapshot: do not inspect its churn labels.
HELD_OUT = pd.Timestamp("2013-10-01")


def make_window(snapshot):
    return {
        "observation_start": snapshot - pd.DateOffset(
            months=HISTORY_MONTHS
        ),
        "observation_end": snapshot,
        "label_start": snapshot,
        "label_end": snapshot + pd.DateOffset(
            months=LABEL_MONTHS
        ),
    }


def ready_before(snapshot):
    """Earlier snapshots whose labels are already known."""
    return [
        s for s in SNAPSHOTS
        if s < snapshot
        and s + pd.DateOffset(months=LABEL_MONTHS)
        <= snapshot
    ]


def valid_validation_dates(test_snapshot):
    """Validation dates usable before the final test."""
    return [
        s for s in SNAPSHOTS
        if s < test_snapshot
        and s + pd.DateOffset(months=LABEL_MONTHS)
        <= test_snapshot
        and len(ready_before(s)) > 0
    ]


def main():
    orders = pd.read_csv(
        DATA_PATH,
        parse_dates=["OrderDate"],
    )

    first_date = orders["OrderDate"].min()
    last_date = orders["OrderDate"].max()

    print("=== Stage 1D.3: Rolling Window Audit ===")
    print(f"Data coverage: {first_date.date()} -> {last_date.date()}")

    print("\nSNAPSHOT AVAILABILITY")
    print("-" * 75)
    print(
        f"{'Snapshot':<13} {'Eligible':>9} "
        f"{'Churners':>10} {'Rate':>9} "
        f"{'Ready train snapshots':>23}"
    )

    results = {}

    for snapshot in SNAPSHOTS:
        window = make_window(snapshot)

        assert window["observation_start"] >= first_date
        assert window["label_end"] <= last_date

        # Eligibility: at least one order in previous 6 months.
        recent_start = snapshot - pd.DateOffset(months=6)

        eligible = orders.loc[
            (orders["OrderDate"] >= recent_start)
            & (orders["OrderDate"] < snapshot),
            "StoreID",
        ].unique()

        ready = ready_before(snapshot)

        if snapshot == HELD_OUT:
            # Keep potential final test labels hidden.
            churners = "HIDDEN"
            rate = "HIDDEN"
        else:
            labels = build_labels(orders, window)

            assert len(labels) == len(eligible)
            assert set(labels["StoreID"]) == set(eligible)

            n_churn = int(labels["churn"].sum())
            churners = str(n_churn)
            rate = f"{labels['churn'].mean() * 100:.2f}%"

        results[snapshot] = len(eligible)

        print(
            f"{snapshot.date()!s:<13} "
            f"{len(eligible):>9} "
            f"{churners:>10} "
            f"{rate:>9} "
            f"{len(ready):>23}"
        )

    # Confirm original Stage 1C results.
    assert results[pd.Timestamp("2012-07-01")] == 326
    assert results[pd.Timestamp("2013-01-01")] == 343
    assert results[pd.Timestamp("2013-07-01")] == 445

    print("\nSPLIT STRATEGY COMPARISON")
    print("-" * 75)

    test_candidates = [
        pd.Timestamp("2013-07-01"),
        pd.Timestamp("2013-10-01"),
    ]

    for test_date in test_candidates:
        validation_dates = valid_validation_dates(test_date)

        print(f"\nFinal test snapshot: {test_date.date()}")
        print(f"Available validation dates: {len(validation_dates)}")

        for val_date in validation_dates:
            train_dates = ready_before(val_date)

            print(
                f"  Validation {val_date.date()} "
                f"| {len(train_dates)} ready training snapshots"
            )

    print("\n[PASS] Rolling window audit completed")


if __name__ == "__main__":
    main()

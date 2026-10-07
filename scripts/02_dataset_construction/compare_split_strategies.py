import pandas as pd

from check_churn_labels import build_labels


DATA_PATH = "datasets/orders_clean.csv"

TEST_DATE = pd.Timestamp("2013-10-01")

FINAL_TRAIN_DATES = [
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
]

STRATEGIES = {
    "fixed": [
        {
            "train": ["2012-07-01"],
            "validation": "2013-01-01",
        },
    ],
    "rolling": [
        {
            "train": ["2012-07-01"],
            "validation": "2013-01-01",
        },
        {
            "train": ["2012-07-01", "2012-10-01"],
            "validation": "2013-04-01",
        },
    ],
}


def make_window(date):
    t = pd.Timestamp(date)

    return {
        "observation_start": t - pd.DateOffset(months=12),
        "observation_end": t,
        "label_start": t,
        "label_end": t + pd.DateOffset(months=6),
    }


def get_labels(orders, date):
    return build_labels(orders, make_window(date))


def check_chronology(train_dates, validation_date):
    validation = pd.Timestamp(validation_date)

    for date in train_dates:
        train = pd.Timestamp(date)

        assert train < validation
        assert train + pd.DateOffset(months=6) <= validation

    assert validation + pd.DateOffset(months=6) <= TEST_DATE


def report(name, df):
    churners = int(df["churn"].sum())
    rate = df["churn"].mean() * 100

    print(
        f"{name:<15} "
        f"Rows: {len(df):>4} | "
        f"Churners: {churners:>3} | "
        f"Rate: {rate:>6.2f}%"
    )


def main():
    orders = pd.read_csv(
        DATA_PATH,
        parse_dates=["OrderDate"],
    )

    print("=== Stage 1D.4: Strategy Comparison ===")

    all_dates = sorted({
        date
        for folds in STRATEGIES.values()
        for fold in folds
        for date in (
            fold["train"] + [fold["validation"]]
        )
    } | set(FINAL_TRAIN_DATES))

    # Generate labels only for historical training
    # and validation snapshots, never final test.
    labels = {}

    for date in all_dates:
        window = make_window(date)

        assert window["observation_start"] >= orders["OrderDate"].min()
        assert window["label_end"] <= orders["OrderDate"].max()
        assert window["label_end"] <= TEST_DATE

        labels[date] = get_labels(orders, date)

    summary = []

    for strategy, folds in STRATEGIES.items():
        print(f"\n{'=' * 65}")
        print(strategy.upper())
        print("=" * 65)

        validation_parts = []

        for i, fold in enumerate(folds, start=1):
            check_chronology(
                fold["train"],
                fold["validation"],
            )

            train = pd.concat(
                [labels[d] for d in fold["train"]],
                ignore_index=True,
            )

            validation = labels[fold["validation"]]

            print(f"\nFold {i}")
            print(f"Train dates: {fold['train']}")
            print(f"Validation: {fold['validation']}")

            report("Train", train)
            report("Validation", validation)

            validation_parts.append(validation)

        combined = pd.concat(
            validation_parts,
            ignore_index=True,
        )

        summary.append({
            "strategy": strategy,
            "folds": len(folds),
            "validation_rows": len(combined),
            "validation_churners": int(combined["churn"].sum()),
            "validation_rate": round(
                combined["churn"].mean() * 100, 2
            ),
        })

        print("\n[PASS] Chronological checks")

    # Both approaches use identical final training data.
    final_train = pd.concat(
        [labels[d] for d in FINAL_TRAIN_DATES],
        ignore_index=True,
    )

    for date in FINAL_TRAIN_DATES:
        assert (
            pd.Timestamp(date) + pd.DateOffset(months=6)
            <= TEST_DATE
        )

    # Count final test eligibility without reading labels.
    test_start = TEST_DATE - pd.DateOffset(months=6)

    test_eligible = orders.loc[
        (orders["OrderDate"] >= test_start)
        & (orders["OrderDate"] < TEST_DATE),
        "StoreID",
    ].nunique()

    assert test_eligible == 492

    print("\n=== Strategy Summary ===")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n=== Shared Final Dataset ===")
    print(f"Training observations: {len(final_train)}")
    print(f"Test eligible resellers: {test_eligible}")
    print("Test labels: NOT ACCESSED")

    print("\n[PASS] Both strategies verified")


if __name__ == "__main__":
    main()

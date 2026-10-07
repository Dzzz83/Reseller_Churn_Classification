
import pandas as pd

from define_windows import WINDOWS, select_period


DATA_PATH = "datasets/orders_clean.csv"

EXPECTED = {
    "train": (326, 76),
    "validation": (343, 25),
    "test": (445, 83),
}


def investigate(orders, split, window):
    obs_end = pd.Timestamp(window["observation_end"])
    label_start = pd.Timestamp(window["label_start"])
    label_end = pd.Timestamp(window["label_end"])

    # Same 6-month eligibility rule as Stage 1C
    history = select_period(
        orders,
        obs_end - pd.DateOffset(months=6),
        obs_end,
    )

    eligible_ids = pd.Index(
        sorted(history["StoreID"].unique())
    )

    # Only future purchases from eligible resellers
    future = select_period(orders, label_start, label_end)
    eligible_future = future[
        future["StoreID"].isin(eligible_ids)
    ].copy()

    # Count future orders per reseller
    future_counts = (
        eligible_future.groupby("StoreID")
        .size()
        .reindex(eligible_ids, fill_value=0)
    )

    churners = int((future_counts == 0).sum())
    churn_rate = churners / len(eligible_ids) * 100

    # Verify consistency with Stage 1D
    assert (len(eligible_ids), churners) == EXPECTED[split]

    print(f"\n{'=' * 55}")
    print(split.upper())
    print("=" * 55)

    print(f"Eligible resellers: {len(eligible_ids)}")
    print(f"Churners:           {churners}")
    print(f"Churn rate:         {churn_rate:.2f}%")

    # Future purchasing frequency
    print("\nPurchasing frequency during label period")
    print("-" * 45)
    print(f"0 orders:  {(future_counts == 0).sum()}")
    print(f"1 order:   {(future_counts == 1).sum()}")
    print(f"2 orders:  {(future_counts == 2).sum()}")
    print(f"3+ orders: {(future_counts >= 3).sum()}")

    assert len(future_counts) == len(eligible_ids)
    assert (
        (future_counts == 0).sum()
        + (future_counts == 1).sum()
        + (future_counts == 2).sum()
        + (future_counts >= 3).sum()
        == len(eligible_ids)
    )

    # Monthly purchasing activity
    print("\nMonthly purchasing activity")
    print("-" * 55)
    print(
        f"{'Month':<10} {'All orders':>12} "
        f"{'All stores':>12} {'Eligible buyers':>16}"
    )

    months = pd.period_range(
        label_start,
        label_end - pd.Timedelta(days=1),
        freq="M",
    )

    for month in months:
        month_orders = future[
            future["OrderDate"].dt.to_period("M") == month
        ]

        month_eligible = eligible_future[
            eligible_future["OrderDate"].dt.to_period("M")
            == month
        ]

        print(
            f"{str(month):<10} "
            f"{len(month_orders):>12} "
            f"{month_orders['StoreID'].nunique():>12} "
            f"{month_eligible['StoreID'].nunique():>16}"
        )

    print("\n[PASS] Cohort counts and labels verified")


def main():
    orders = pd.read_csv(
        DATA_PATH,
        parse_dates=["OrderDate"],
    )

    print("=== Stage 1D.1: Churn Distribution Investigation ===")

    for split, window in WINDOWS.items():
        investigate(orders, split, window)

    print("\n[PASS] Investigation completed")


if __name__ == "__main__":
    main()


import pandas as pd

from define_windows import WINDOWS, select_period


DATA_PATH = "datasets/orders_clean.csv"


def build_labels(orders, window):
    obs_end = pd.Timestamp(window["observation_end"])
    label_start = pd.Timestamp(window["label_start"])
    label_end = pd.Timestamp(window["label_end"])

    # Only resellers active in the previous 6 months
    recent_orders = select_period(
        orders,
        obs_end - pd.DateOffset(months=6),
        obs_end,
    )

    eligible_ids = sorted(recent_orders["StoreID"].unique())

    # Purchases in the following 6 months
    future_orders = select_period(
        orders, label_start, label_end
    )

    future_counts = future_orders.groupby("StoreID").size()

    labels = pd.DataFrame({"StoreID": eligible_ids})
    labels["future_orders"] = (
        labels["StoreID"].map(future_counts).fillna(0).astype(int)
    )

    labels["churn"] = (labels["future_orders"] == 0).astype(int)

    return labels


def main():
    orders = pd.read_csv(DATA_PATH, parse_dates=["OrderDate"])

    print("=== Stage 1D: Churn Label Verification ===")

    for split, window in WINDOWS.items():
        labels = build_labels(orders, window)

        churners = int(labels["churn"].sum())
        non_churners = len(labels) - churners
        churn_rate = labels["churn"].mean() * 100

        print(f"\n{split.upper()}")
        print("-" * 45)
        print(f"Eligible resellers: {len(labels)}")
        print(f"Churners (1):       {churners}")
        print(f"Non-churners (0):   {non_churners}")
        print(f"Churn rate:         {churn_rate:.2f}%")

        # Verify the label definition
        assert labels["StoreID"].is_unique
        assert labels["churn"].isin([0, 1]).all()
        assert (
            (labels["churn"] == 1)
            == (labels["future_orders"] == 0)
        ).all()

        # Both classes are needed for binary classification
        if churners == 0 or non_churners == 0:
            print("[WARNING] Only one class is present.")
        else:
            print("[PASS] Both churn classes are present")

        print("\nSample labels:")
        for churn_value in (0, 1):
            sample = labels[labels["churn"] == churn_value].head(3)
            print(sample.to_string(index=False))

        print("[PASS] Label consistency checks")

    print("\nStage 1D label audit completed.")


if __name__ == "__main__":
    main()

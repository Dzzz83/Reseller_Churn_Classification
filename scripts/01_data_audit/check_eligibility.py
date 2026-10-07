
import pandas as pd

from define_windows import WINDOWS, select_period


DATA_PATH = "datasets/orders_clean.csv"


def check_eligibility(orders, split, window):
    obs_start = pd.Timestamp(window["observation_start"])
    obs_end = pd.Timestamp(window["observation_end"])

    # Approach A: At least one order in 12 months
    obs_12m = select_period(orders, obs_start, obs_end)
    eligible_12m = set(obs_12m["StoreID"])

    # Approach B: At least one order in last 6 months
    last_6m_start = obs_end - pd.DateOffset(months=6)
    obs_6m = select_period(orders, last_6m_start, obs_end)
    eligible_6m = set(obs_6m["StoreID"])

    # Resellers excluded by the 6-month rule
    excluded = eligible_12m - eligible_6m

    retention = (
        len(eligible_6m) / len(eligible_12m) * 100
        if eligible_12m else 0
    )

    print(f"\n{split.upper()}")
    print("-" * 45)
    print(f"12-month eligible: {len(eligible_12m)}")
    print(f"6-month eligible:  {len(eligible_6m)}")
    print(f"Excluded:          {len(excluded)}")
    print(f"Retention:         {retention:.1f}%")

    # Validation checks
    assert eligible_6m.issubset(eligible_12m)
    assert len(eligible_12m) == (
        len(eligible_6m) + len(excluded)
    )

    print("[PASS] Eligibility sets are consistent")

    return eligible_12m, eligible_6m


def main():
    orders = pd.read_csv(DATA_PATH, parse_dates=["OrderDate"])

    print("=== Stage 1C: Eligibility Comparison ===")

    for split, window in WINDOWS.items():
        check_eligibility(orders, split, window)

    print("\n[PASS] Stage 1C eligibility audit completed.")


if __name__ == "__main__":
    main()

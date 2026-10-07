
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("datasets/processed/rfm_recent_gaps_features.csv")
ORDERS = Path("datasets/orders_clean.csv")
OUTPUT = Path("datasets/processed/rfm_recent_gaps_trend_features.csv")

TOLERANCE = 0.01


def calculate_trend(df):
    """Calculate revenue trends from two consecutive 6-month periods."""
    result = df.copy()

    # Revenue from the older 6 months.
    previous = result["revenue_12m"] - result["revenue_6m"]

    assert (previous >= -TOLERANCE).all()
    previous = previous.clip(lower=0)

    recent = result["revenue_6m"]

    # Distinguish established purchasing history from new activity.
    has_history = previous > TOLERANCE

    result["has_previous_6m_revenue"] = has_history.astype(int)

    # Revenue trend is undefined without previous-period revenue.
    result["revenue_trend"] = np.where(
        has_history,
        np.log1p(recent) - np.log1p(previous),
        np.nan,
    )

    return result


def verify(features, original, orders):
    # 1. Original features must remain unchanged.
    pd.testing.assert_frame_equal(
        features[original.columns],
        original,
    )

    assert len(features) == len(original)
    assert len(features) == 2312
    assert not features.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    # 2. Verify history indicator and missing-value rules.
    previous = (
        features["revenue_12m"] - features["revenue_6m"]
    ).clip(lower=0)

    expected_history = previous > TOLERANCE
    has_history = features["has_previous_6m_revenue"] == 1

    assert (has_history == expected_history).all()

    # Missing trends must correspond exactly to missing history.
    assert features.loc[
        ~has_history, "revenue_trend"
    ].isna().all()

    assert features.loc[
        has_history, "revenue_trend"
    ].notna().all()

    assert np.isfinite(
        features.loc[has_history, "revenue_trend"]
    ).all()

    print("[PASS] History indicator and missing-value rules")

    # 3. Verify synthetic edge cases.
    examples = pd.DataFrame({
        "revenue_12m": [300, 300, 200, 100, 0],
        "revenue_6m":  [200, 100, 100, 100, 0],
    })

    calculated = calculate_trend(examples)
    scores = calculated["revenue_trend"]
    flags = calculated["has_previous_6m_revenue"]

    assert scores.iloc[0] > 0           # Increasing
    assert scores.iloc[1] < 0           # Decreasing
    assert np.isclose(scores.iloc[2], 0) # Unchanged

    assert pd.isna(scores.iloc[3])       # No previous revenue
    assert pd.isna(scores.iloc[4])       # No revenue in either period

    assert flags.tolist() == [1, 1, 1, 0, 0]

    print("[PASS] Synthetic edge cases")

    # 4. Independently verify three real resellers.
    snapshot = pd.Timestamp("2012-07-01")
    midpoint = snapshot - pd.DateOffset(months=6)
    start = snapshot - pd.DateOffset(months=12)

    for store_id in (326, 292, 616):
        history = orders[
            (orders["StoreID"] == store_id)
            & (orders["OrderDate"] >= start)
            & (orders["OrderDate"] < snapshot)
        ]

        previous_revenue = history.loc[
            history["OrderDate"] < midpoint,
            "SubTotal",
        ].sum()

        recent_revenue = history.loc[
            history["OrderDate"] >= midpoint,
            "SubTotal",
        ].sum()

        row = features.loc[
            (features["StoreID"] == store_id)
            & (features["snapshot"] == snapshot)
        ].iloc[0]

        actual = row["revenue_trend"]
        actual_flag = row["has_previous_6m_revenue"]

        if previous_revenue <= TOLERANCE:
            assert pd.isna(actual)
            assert actual_flag == 0
        else:
            expected = (
                np.log1p(recent_revenue)
                - np.log1p(previous_revenue)
            )

            assert np.isclose(actual, expected, atol=1e-9)
            assert actual_flag == 1

    print("[PASS] Three reseller calculations")
    print("[PASS] Existing features preserved")
    print("[PASS] No invalid or infinite trend values")


def main():
    original = pd.read_csv(
        INPUT,
        parse_dates=["snapshot"],
    )

    orders = pd.read_csv(
        ORDERS,
        parse_dates=["OrderDate"],
    )

    print("=== Stage 1E.6: Revenue Trend ===\n")

    features = calculate_trend(original)
    verify(features, original, orders)

    trend = features["revenue_trend"]

    print("\nRevenue trend distribution:")
    print(trend.describe().round(3))

    print("\nTrend categories:")
    print(f"Increasing: {(trend > 1e-10).sum()}")
    print(f"Decreasing: {(trend < -1e-10).sum()}")
    print(f"Unchanged:  {(trend.abs() <= 1e-10).sum()}")
    print(f"Undefined:  {trend.isna().sum()}")

    no_history = (
        features["has_previous_6m_revenue"] == 0
    ).sum()

    print(f"\nNo previous 6-month revenue: {no_history}")
    print(f"Total observations: {len(features)}")

    print("\nMissing values in new features:")
    print(
        features[
            ["has_previous_6m_revenue", "revenue_trend"]
        ].isna().sum()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"\nSaved: {OUTPUT}")
    print("\n[PASS] Stage 1E.6 completed")


if __name__ == "__main__":
    main()

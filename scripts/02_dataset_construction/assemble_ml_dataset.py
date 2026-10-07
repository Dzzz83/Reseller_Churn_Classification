
from pathlib import Path

import pandas as pd

from check_churn_labels import build_labels
from compare_split_strategies import make_window


DATA_DIR = Path("datasets/processed")
FEATURES_PATH = DATA_DIR / "reseller_order_age_features.csv"

TRAIN_DATES = [
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
]

TEST_DATE = "2013-10-01"

EXPECTED_COUNTS = {
    "2012-07-01": 326,
    "2012-10-01": 366,
    "2013-01-01": 343,
    "2013-04-01": 340,
}


def main():
    print("=== Stage 1H.1: Assemble ML Dataset ===\n")

    orders = pd.read_csv(
        "datasets/orders_clean.csv",
        parse_dates=["OrderDate"],
    )

    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["snapshot"],
    )

    assert not features.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    labeled_parts = []

    # Build labels only for historical snapshots.
    for date in TRAIN_DATES:
        snapshot = pd.Timestamp(date)

        # Labels must be known before final test.
        assert (
            snapshot + pd.DateOffset(months=6)
            <= pd.Timestamp(TEST_DATE)
        )

        X = features[
            features["snapshot"] == snapshot
        ].copy()

        labels = build_labels(
            orders,
            make_window(date),
        )

        assert len(X) == EXPECTED_COUNTS[date]
        assert len(labels) == len(X)
        assert set(X["StoreID"]) == set(labels["StoreID"])

        # Join only the target, never future_orders.
        result = X.merge(
            labels[["StoreID", "churn"]],
            on="StoreID",
            how="left",
            validate="one_to_one",
        )

        assert result["churn"].notna().all()
        assert result["churn"].isin([0, 1]).all()

        result["churn"] = result["churn"].astype(int)

        labeled_parts.append(result)

        print(
            f"[PASS] {date}: "
            f"{len(result)} observations, "
            f"{result['churn'].sum()} churners"
        )

    train = pd.concat(
        labeled_parts,
        ignore_index=True,
    )

    # Final test: features only.
    test = features[
        features["snapshot"] == pd.Timestamp(TEST_DATE)
    ].copy()

    assert len(train) == 1375
    assert len(test) == 492

    assert "churn" not in test.columns
    assert "future_orders" not in train.columns
    assert "future_orders" not in test.columns

    assert not train.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    assert not test.duplicated(
        ["StoreID", "snapshot"]
    ).any()

    assert (
        train["snapshot"].max() < test["snapshot"].min()
    )

    # Ensure the feature schema is identical.
    assert list(train.drop(columns="churn")) == list(test)

    # Save datasets.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        DATA_DIR / "ml_labeled_snapshots.csv",
        index=False,
    )

    test.to_csv(
        DATA_DIR / "ml_test_features.csv",
        index=False,
    )

    print("\nDataset summary")
    print("-" * 45)

    print(f"Labeled observations: {len(train)}")
    print(f"Churners:             {train['churn'].sum()}")
    print(f"Non-churners:         {(train['churn'] == 0).sum()}")
    print(f"Test observations:    {len(test)}")
    print("Test labels:          NOT ACCESSED")

    print("\n[PASS] ML dataset assembly completed")


if __name__ == "__main__":
    main()

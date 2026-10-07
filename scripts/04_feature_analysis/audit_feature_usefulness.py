from pathlib import Path

import pandas as pd

from define_feature_sets import FEATURE_SETS


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

TRAIN_DATES = [
    "2012-07-01",
    "2012-10-01",
]


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    # Historical training observations only.
    train = df[
        df["snapshot"].isin(
            pd.to_datetime(TRAIN_DATES)
        )
    ].copy()

    features = FEATURE_SETS["full"]

    print("=== Feature Usefulness Audit ===\n")

    print(f"Training observations: {len(train)}")
    print(f"Churners: {train['churn'].sum()}")
    print(
        f"Non-churners: "
        f"{(train['churn'] == 0).sum()}"
    )

    assert len(train) == 692
    assert train["churn"].isin([0, 1]).all()

    results = []

    for feature in features:
        non_churn = train.loc[
            train["churn"] == 0,
            feature,
        ]

        churn = train.loc[
            train["churn"] == 1,
            feature,
        ]

        # Spearman handles skewed variables better
        # than Pearson for this exploratory audit.
        rho = train[[feature, "churn"]].corr(
            method="spearman"
        ).loc[feature, "churn"]

        results.append({
            "feature": feature,

            "non_churn_median":
                non_churn.median(),

            "churn_median":
                churn.median(),

            "median_difference":
                churn.median()
                - non_churn.median(),

            "spearman_rho": rho,

            "abs_rho":
                abs(rho)
                if pd.notna(rho)
                else 0,

            "non_churn_missing_pct":
                non_churn.isna().mean() * 100,

            "churn_missing_pct":
                churn.isna().mean() * 100,
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "abs_rho",
        ascending=False,
    )

    print("\nFeature relationship with churn")
    print("-" * 110)

    display_columns = [
        "feature",
        "non_churn_median",
        "churn_median",
        "spearman_rho",
        "non_churn_missing_pct",
        "churn_missing_pct",
    ]

    print(
        results[display_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nTop features by absolute Spearman correlation")
    print("-" * 60)

    for _, row in results.head(10).iterrows():
        direction = (
            "higher → more churn"
            if row["spearman_rho"] > 0
            else "higher → less churn"
        )

        print(
            f"{row['feature']:<28} "
            f"rho={row['spearman_rho']:>7.3f} "
            f"{direction}"
        )

    print("\n[PASS] Feature usefulness audit completed")
    print("Validation and final test data were not used.")


if __name__ == "__main__":
    main()
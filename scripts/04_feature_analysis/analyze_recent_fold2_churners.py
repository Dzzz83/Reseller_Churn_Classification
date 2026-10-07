from pathlib import Path

import numpy as np
import pandas as pd


ORDERS_PATH = Path("datasets/orders_clean.csv")
DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path("results/error_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2013-04-01")
RECENT_DAYS = 30


def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def build_history_features(orders, store_id):
    history = orders[
        (orders["StoreID"] == store_id)
        & (orders["OrderDate"] < SNAPSHOT)
    ].sort_values(
        ["OrderDate", "SalesOrderID"]
    )

    dates = (
        history["OrderDate"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    gaps = dates.diff().dt.days.dropna()

    history_12m = history[
        history["OrderDate"]
        >= SNAPSHOT - pd.DateOffset(months=12)
    ]

    history_6m = history[
        history["OrderDate"]
        >= SNAPSHOT - pd.DateOffset(months=6)
    ]

    history_3m = history[
        history["OrderDate"]
        >= SNAPSHOT - pd.DateOffset(months=3)
    ]

    previous_3m = history[
        (
            history["OrderDate"]
            >= SNAPSHOT - pd.DateOffset(months=6)
        )
        & (
            history["OrderDate"]
            < SNAPSHOT - pd.DateOffset(months=3)
        )
    ]

    previous_6m = history[
        (
            history["OrderDate"]
            >= SNAPSHOT - pd.DateOffset(months=12)
        )
        & (
            history["OrderDate"]
            < SNAPSHOT - pd.DateOffset(months=6)
        )
    ]

    mean_gap = gaps.mean()
    std_gap = gaps.std(ddof=1)
    max_gap = gaps.max()

    last_order = history.iloc[-1]

    second_last_recency = np.nan
    if len(history) >= 2:
        second_last_recency = (
            SNAPSHOT - history.iloc[-2]["OrderDate"]
        ).days

    active_months_12m = (
        history_12m["OrderDate"]
        .dt.to_period("M")
        .nunique()
    )

    active_months_6m = (
        history_6m["OrderDate"]
        .dt.to_period("M")
        .nunique()
    )

    revenue_3m = history_3m["SubTotal"].sum()
    revenue_prev3m = previous_3m["SubTotal"].sum()

    revenue_6m = history_6m["SubTotal"].sum()
    revenue_prev6m = previous_6m["SubTotal"].sum()

    average_order_value_12m = (
        history_12m["SubTotal"].mean()
    )

    return {
        "n_orders_total_before_snapshot":
            history["SalesOrderID"].nunique(),

        "tenure_days":
            (
                history["OrderDate"].max()
                - history["OrderDate"].min()
            ).days,

        "active_months_12m":
            active_months_12m,

        "active_months_6m":
            active_months_6m,

        "second_last_recency_days":
            second_last_recency,

        "last_gap_days":
            gaps.iloc[-1]
            if len(gaps) > 0
            else np.nan,

        "mean_gap_all":
            mean_gap,

        "std_gap_all":
            std_gap,

        "gap_cv_all":
            safe_divide(
                std_gap,
                mean_gap,
            ),

        "max_gap_all":
            max_gap,

        "gap_120plus_count":
            int((gaps >= 120).sum()),

        "gap_150plus_count":
            int((gaps >= 150).sum()),

        "gap_120plus_fraction":
            (
                (gaps >= 120).mean()
                if len(gaps) > 0
                else np.nan
            ),

        "max_gap_to_mean_gap":
            safe_divide(
                max_gap,
                mean_gap,
            ),

        "current_recency_to_mean_gap":
            safe_divide(
                (
                    SNAPSHOT
                    - last_order["OrderDate"]
                ).days,
                mean_gap,
            ),

        "n_orders_6m":
            history_6m[
                "SalesOrderID"
            ].nunique(),

        "n_orders_prev6m":
            previous_6m[
                "SalesOrderID"
            ].nunique(),

        "orders_6m_change":
            (
                history_6m[
                    "SalesOrderID"
                ].nunique()
                - previous_6m[
                    "SalesOrderID"
                ].nunique()
            ),

        "n_orders_3m":
            history_3m[
                "SalesOrderID"
            ].nunique(),

        "n_orders_prev3m":
            previous_3m[
                "SalesOrderID"
            ].nunique(),

        "orders_3m_change":
            (
                history_3m[
                    "SalesOrderID"
                ].nunique()
                - previous_3m[
                    "SalesOrderID"
                ].nunique()
            ),

        "revenue_6m_log_change":
            (
                np.log1p(revenue_6m)
                - np.log1p(
                    revenue_prev6m
                )
            ),

        "revenue_3m_log_change":
            (
                np.log1p(revenue_3m)
                - np.log1p(
                    revenue_prev3m
                )
            ),

        "latest_order_value_ratio_12m":
            safe_divide(
                last_order["SubTotal"],
                average_order_value_12m,
            ),
    }


def separation_auc(
    churn_values,
    nonchurn_values,
):
    churn_values = (
        pd.Series(churn_values)
        .dropna()
        .to_numpy()
    )

    nonchurn_values = (
        pd.Series(nonchurn_values)
        .dropna()
        .to_numpy()
    )

    if (
        len(churn_values) == 0
        or len(nonchurn_values) == 0
    ):
        return np.nan, ""

    wins = 0.0
    total = 0

    for churn_value in churn_values:
        for nonchurn_value in nonchurn_values:
            total += 1

            if churn_value > nonchurn_value:
                wins += 1
            elif churn_value == nonchurn_value:
                wins += 0.5

    raw_auc = wins / total

    if raw_auc >= 0.5:
        return (
            raw_auc,
            "higher_in_churn",
        )

    return (
        1 - raw_auc,
        "lower_in_churn",
    )


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
        encoding="utf-8-sig",
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    fold2 = df[
        df["snapshot"] == SNAPSHOT
    ].copy()

    recent = fold2[
        fold2["recency_days"]
        <= RECENT_DAYS
    ].copy()

    assert len(recent) == 134
    assert recent["churn"].sum() == 28

    rows = []

    for _, row in recent.iterrows():
        features = build_history_features(
            orders,
            row["StoreID"],
        )

        rows.append(
            {
                "StoreID":
                    row["StoreID"],

                "churn":
                    int(row["churn"]),

                **features,
            }
        )

    analysis = pd.DataFrame(rows)

    churn = analysis[
        analysis["churn"] == 1
    ]

    nonchurn = analysis[
        analysis["churn"] == 0
    ]

    feature_columns = [
        column
        for column
        in analysis.columns
        if column
        not in {
            "StoreID",
            "churn",
        }
    ]

    summary_rows = []

    for feature in feature_columns:
        auc, direction = separation_auc(
            churn[feature],
            nonchurn[feature],
        )

        summary_rows.append(
            {
                "feature":
                    feature,

                "churn_median":
                    churn[
                        feature
                    ].median(),

                "nonchurn_median":
                    nonchurn[
                        feature
                    ].median(),

                "churn_mean":
                    churn[
                        feature
                    ].mean(),

                "nonchurn_mean":
                    nonchurn[
                        feature
                    ].mean(),

                "separation_auc":
                    auc,

                "direction":
                    direction,

                "churn_nonmissing":
                    churn[
                        feature
                    ].notna().sum(),

                "nonchurn_nonmissing":
                    nonchurn[
                        feature
                    ].notna().sum(),
            }
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "separation_auc",
            ascending=False,
        )
    )

    # Prior churn-like behavior from already-known
    # historical snapshots only.
    prior_dates = pd.to_datetime(
        [
            "2012-07-01",
            "2012-10-01",
            "2013-01-01",
        ]
    )

    prior = df[
        df["snapshot"].isin(
            prior_dates
        )
    ][
        [
            "StoreID",
            "snapshot",
            "churn",
        ]
    ]

    prior_counts = (
        prior.groupby(
            "StoreID"
        )["churn"]
        .sum()
        .rename(
            "prior_churn_labels"
        )
    )

    analysis = analysis.merge(
        prior_counts,
        on="StoreID",
        how="left",
    )

    analysis[
        "prior_churn_labels"
    ] = (
        analysis[
            "prior_churn_labels"
        ]
        .fillna(0)
        .astype(int)
    )

    analysis[
        "had_prior_churn_label"
    ] = (
        analysis[
            "prior_churn_labels"
        ]
        > 0
    ).astype(int)

    analysis.to_csv(
        RESULTS_DIR
        / "fold2_recent30_resellers.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR
        / "fold2_recent30_behavior_comparison.csv",
        index=False,
    )

    print(
        "=== Fold-2 Recent Buyer "
        "Behavior Analysis ==="
    )

    print(
        f"Recent buyers:      "
        f"{len(analysis)}"
    )

    print(
        f"Recent churners:    "
        f"{analysis['churn'].sum()}"
    )

    print(
        f"Recent non-churners:"
        f" {(analysis['churn'] == 0).sum()}"
    )

    print()
    print(
        "=== Strongest Historical "
        "Behavior Differences ==="
    )

    print(
        summary.head(12).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=== Prior Churn-Like "
        "Behavior ==="
    )

    for value in [
        0,
        1,
    ]:
        group = analysis[
            analysis["churn"]
            == value
        ]

        label = (
            "Churn"
            if value == 1
            else "Non-churn"
        )

        prior_rate = (
            group[
                "had_prior_churn_label"
            ].mean()
        )

        long_gap_rate = (
            group[
                "gap_120plus_count"
            ]
            .gt(0)
            .mean()
        )

        print(
            f"{label}: "
            f"prior churn episode="
            f"{prior_rate:.2%}, "
            f"prior 120+ day gap="
            f"{long_gap_rate:.2%}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "- A separation AUC near 0.50 "
        "means the feature barely separates "
        "the two groups."
    )
    print(
        "- These recent churners do not show "
        "stronger historical irregularity."
    )
    print(
        "- Final-test labels were NOT used."
    )


if __name__ == "__main__":
    main()

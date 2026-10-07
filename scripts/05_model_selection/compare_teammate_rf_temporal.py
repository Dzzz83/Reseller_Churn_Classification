from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ORDERS_PATH = Path("datasets/orders_clean.csv")
STORES_PATH = Path("datasets/stores_clean.csv")
ML_PATH = Path("datasets/processed/ml_labeled_snapshots.csv")

RESULTS_DIR = Path("results/model_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


FOLDS = {
    "fold_1": {
        "train": ["2012-07-01"],
        "validation": "2013-01-01",
    },
    "fold_2": {
        "train": [
            "2012-07-01",
            "2012-10-01",
        ],
        "validation": "2013-04-01",
    },
}


SEEDS = [42, 78, 88, 1034, 2026]


# Our current locked feature set.
OUR_FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]


OUR_RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


# Leakage-safe features inspired by teammate's full-history model.
TEAM_NUMERIC = [
    "recency_days",
    "tenure_days",
    "n_orders_total",
    "revenue_total",
    "aov",
    "avg_qty",
    "avg_lines",
    "avg_disc",
    "n_orders_6m",
    "revenue_6m",
    "n_orders_12m",
    "revenue_12m",
    "rev_trend",
    "mean_gap",
    "std_gap",
    "overdue_ratio",
    "share_bikes",
    "share_components",
    "share_clothing",
    "share_accessories",
    "store_age",
]

TEAM_CATEGORICAL = [
    "TerritoryID",
]

TEAM_FEATURES = TEAM_NUMERIC + TEAM_CATEGORICAL


TEAM_LOG = [
    "recency_days",
    "tenure_days",
    "n_orders_total",
    "revenue_total",
    "aov",
    "avg_qty",
    "avg_lines",
    "n_orders_6m",
    "revenue_6m",
    "n_orders_12m",
    "revenue_12m",
    "rev_trend",
    "mean_gap",
    "std_gap",
]


def detect_store_id(df):
    for col in [
        "StoreID",
        "store_id",
        "ResellerID",
        "reseller_id",
    ]:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find reseller ID column."
    )


def build_teammate_features(
    orders,
    stores,
    snapshot,
    store_ids,
):
    t = pd.Timestamp(snapshot)

    # CRITICAL:
    # only transactions before the snapshot.
    hist = orders[
        orders["OrderDate"] < t
    ].copy()

    hist = hist[
        hist["StoreID"].isin(store_ids)
    ].sort_values(
        ["StoreID", "OrderDate"]
    )

    g = hist.groupby("StoreID")

    f = pd.DataFrame(
        index=pd.Index(
            store_ids,
            name="StoreID",
        )
    )

    f["recency_days"] = (
        t - g["OrderDate"].max()
    ).dt.days

    f["tenure_days"] = (
        t - g["OrderDate"].min()
    ).dt.days

    f["n_orders_total"] = g.size()
    f["revenue_total"] = g["SubTotal"].sum()
    f["aov"] = g["SubTotal"].mean()
    f["avg_qty"] = g["qty"].mean()
    f["avg_lines"] = g["n_lines"].mean()
    f["avg_disc"] = g["avg_disc"].mean()

    # 6-month and 12-month activity.
    for months in [6, 12]:
        window = hist[
            hist["OrderDate"]
            >= t - pd.DateOffset(months=months)
        ]

        wg = window.groupby("StoreID")

        f[f"n_orders_{months}m"] = (
            wg.size()
        )

        f[f"revenue_{months}m"] = (
            wg["SubTotal"].sum()
        )

    f[
        [
            "n_orders_6m",
            "revenue_6m",
            "n_orders_12m",
            "revenue_12m",
        ]
    ] = f[
        [
            "n_orders_6m",
            "revenue_6m",
            "n_orders_12m",
            "revenue_12m",
        ]
    ].fillna(0)

    # Teammate-style trend:
    # recent 6m revenue / previous 6m revenue.
    previous_6m = (
        f["revenue_12m"]
        - f["revenue_6m"]
    )

    f["rev_trend"] = np.where(
        previous_6m > 0,
        f["revenue_6m"] / previous_6m,
        np.nan,
    )

    # Purchase gaps using distinct order dates.
    distinct_dates = hist[
        ["StoreID", "OrderDate"]
    ].drop_duplicates()

    distinct_dates["gap"] = (
        distinct_dates
        .groupby("StoreID")["OrderDate"]
        .diff()
        .dt.days
    )

    gap_group = (
        distinct_dates
        .groupby("StoreID")["gap"]
    )

    f["mean_gap"] = gap_group.mean()
    f["std_gap"] = gap_group.std()

    f["overdue_ratio"] = (
        f["recency_days"]
        / f["mean_gap"]
    )

    # Product mix from historical orders.
    revenue_columns = {
        "bikes": "rev_bikes",
        "components": "rev_components",
        "clothing": "rev_clothing",
        "accessories": "rev_accessories",
    }

    mix = g[
        list(revenue_columns.values())
    ].sum()

    for name, column in revenue_columns.items():
        f[f"share_{name}"] = (
            mix[column]
            / f["revenue_total"]
        )

    # Territory from the most recent
    # PRE-SNAPSHOT transaction.
    f["TerritoryID"] = (
        g["TerritoryID"].last()
    )

    # Only safe store-profile feature.
    store_profile = (
        stores[
            ["StoreID", "YearOpened"]
        ]
        .drop_duplicates("StoreID")
        .set_index("StoreID")
    )

    f = f.join(
        store_profile,
        how="left",
    )

    f["store_age"] = (
        t.year - f["YearOpened"]
    )

    f = f.drop(
        columns=["YearOpened"]
    )

    f.insert(
        0,
        "snapshot",
        t.strftime("%Y-%m-%d"),
    )

    return f.reset_index()


def build_all_teammate_features(
    orders,
    stores,
    ml,
    store_id_col,
):
    rows = []

    historical_snapshots = [
        "2012-07-01",
        "2012-10-01",
        "2013-01-01",
        "2013-04-01",
    ]

    for snapshot in historical_snapshots:

        current = ml[
            ml["snapshot"] == snapshot
        ]

        store_ids = (
            current[store_id_col]
            .astype(int)
            .unique()
        )

        built = build_teammate_features(
            orders,
            stores,
            snapshot,
            store_ids,
        )

        rows.append(built)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_our_model(seed):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    **OUR_RF_PARAMS,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_teammate_model(seed):
    log_features = [
        c
        for c in TEAM_LOG
        if c in TEAM_NUMERIC
    ]

    linear_features = [
        c
        for c in TEAM_NUMERIC
        if c not in log_features
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "log",
                make_pipeline(
                    FunctionTransformer(
                        np.log1p,
                        feature_names_out="one-to-one",
                    ),
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                    StandardScaler(),
                ),
                log_features,
            ),
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                    StandardScaler(),
                ),
                linear_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                TEAM_CATEGORICAL,
            ),
        ]
    )

    # Teammate's reported RF style:
    # 500 trees, min_samples_leaf = 5.
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
        class_weight=None,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(
    name,
    dataset,
    features,
    builder,
):
    rows = []

    for fold_name, fold in FOLDS.items():

        train = dataset[
            dataset["snapshot"].isin(
                fold["train"]
            )
        ]

        validation = dataset[
            dataset["snapshot"]
            == fold["validation"]
        ]

        seed_pr = []
        seed_roc = []

        for seed in SEEDS:
            model = builder(seed)

            model.fit(
                train[features],
                train["churn"],
            )

            probability = (
                model.predict_proba(
                    validation[features]
                )[:, 1]
            )

            seed_pr.append(
                average_precision_score(
                    validation["churn"],
                    probability,
                )
            )

            seed_roc.append(
                roc_auc_score(
                    validation["churn"],
                    probability,
                )
            )

        rows.append(
            {
                "model": name,
                "fold": fold_name,
                "pr_auc_mean": np.mean(seed_pr),
                "pr_auc_std": np.std(seed_pr),
                "roc_auc_mean": np.mean(seed_roc),
                "validation_churn_rate":
                    validation["churn"].mean(),
            }
        )

    return pd.DataFrame(rows)


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["OrderDate"],
    )

    stores = pd.read_csv(
        STORES_PATH,
    )

    ml = pd.read_csv(
        ML_PATH,
    )

    ml["snapshot"] = (
        pd.to_datetime(
            ml["snapshot"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    store_id_col = detect_store_id(ml)

    historical = ml[
        ml["snapshot"].isin(
            [
                "2012-07-01",
                "2012-10-01",
                "2013-01-01",
                "2013-04-01",
            ]
        )
    ].copy()

    print(
        "=== Teammate RF Temporal Comparison ==="
    )
    print(
        "Final test labels: NOT USED"
    )
    print()

    print(
        "Building teammate-style historical features..."
    )

    team_features = (
        build_all_teammate_features(
            orders,
            stores,
            historical,
            store_id_col,
        )
    )

    if store_id_col != "StoreID":
        historical = historical.rename(
            columns={
                store_id_col: "StoreID"
            }
        )

    comparison = historical.merge(
        team_features,
        on=[
            "StoreID",
            "snapshot",
        ],
        how="left",
        suffixes=("", "_team"),
        validate="1:1",
    )

    # Use teammate versions where names overlap.
    for feature in TEAM_FEATURES:
        teammate_name = (
            f"{feature}_team"
        )

        if teammate_name in comparison.columns:
            comparison[feature] = (
                comparison[teammate_name]
            )

    print(
        f"Historical rows: {len(comparison)}"
    )

    print()
    print(
        "Evaluating our tuned Random Forest..."
    )

    our_results = evaluate_model(
        "Our tuned RF",
        historical,
        OUR_FEATURES,
        build_our_model,
    )

    print(
        "Evaluating teammate-style Random Forest..."
    )

    team_results = evaluate_model(
        "Teammate-style RF",
        comparison,
        TEAM_FEATURES,
        build_teammate_model,
    )

    results = pd.concat(
        [
            our_results,
            team_results,
        ],
        ignore_index=True,
    )

    print()
    print(
        "=== Per-Fold Results ==="
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary = (
        results
        .groupby("model")
        .agg(
            mean_pr_auc=(
                "pr_auc_mean",
                "mean",
            ),
            worst_fold_pr_auc=(
                "pr_auc_mean",
                "min",
            ),
            mean_roc_auc=(
                "roc_auc_mean",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_pr_auc",
            ascending=False,
        )
    )

    print()
    print(
        "=== Overall Comparison ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    results.to_csv(
        RESULTS_DIR
        / "teammate_rf_temporal_folds.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR
        / "teammate_rf_temporal_summary.csv",
        index=False,
    )

    print()
    print(
        "Final test labels were NOT used."
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score


ORDERS_PATH = Path("datasets/orders_clean.csv")

OUTPUT_DIR = Path("results/feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR / "targeted_add_one_feature_results.csv"
)

SEEDS = [42, 78, 88, 1034, 2026]


BASE_FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]


CANDIDATES = [
    "avg_order_value_12m",
    "std_order_value_12m",
    "avg_qty_12m",
    "avg_lines_12m",
    "last_order_value",
]


SNAPSHOTS = [
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
]


RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
    "n_jobs": -1,
}


GB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
}


def random_oversample(x, y, seed):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    data = x.copy()
    data["_target"] = y

    non_churn = data[
        data["_target"] == 0
    ]

    churn = data[
        data["_target"] == 1
    ]

    churn_sampled = churn.sample(
        n=len(non_churn),
        replace=True,
        random_state=seed,
    )

    balanced = pd.concat(
        [
            non_churn,
            churn_sampled,
        ],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    y_balanced = balanced.pop(
        "_target"
    ).astype(int)

    return balanced, y_balanced


def build_snapshot(
    orders,
    snapshot_text,
):
    snapshot = pd.Timestamp(
        snapshot_text
    )

    eligibility_start = (
        snapshot
        - pd.DateOffset(months=6)
    )

    history_12m_start = (
        snapshot
        - pd.DateOffset(months=12)
    )

    history_3m_start = (
        snapshot
        - pd.DateOffset(months=3)
    )

    label_end = (
        snapshot
        + pd.DateOffset(months=6)
    )

    pre_snapshot = orders[
        orders["OrderDate"] < snapshot
    ]

    eligible_ids = (
        orders[
            (
                orders["OrderDate"]
                >= eligibility_start
            )
            & (
                orders["OrderDate"]
                < snapshot
            )
        ]["StoreID"]
        .unique()
    )

    history_12m = pre_snapshot[
        (
            pre_snapshot["StoreID"]
            .isin(eligible_ids)
        )
        & (
            pre_snapshot["OrderDate"]
            >= history_12m_start
        )
    ].copy()

    history_3m = pre_snapshot[
        (
            pre_snapshot["StoreID"]
            .isin(eligible_ids)
        )
        & (
            pre_snapshot["OrderDate"]
            >= history_3m_start
        )
    ].copy()

    rows = []

    for store_id in eligible_ids:

        h12 = history_12m[
            history_12m["StoreID"]
            == store_id
        ].sort_values(
            [
                "OrderDate",
                "SalesOrderID",
            ]
        )

        h3 = history_3m[
            history_3m["StoreID"]
            == store_id
        ]

        if h12.empty:
            raise RuntimeError(
                "No 12-month history "
                f"for StoreID {store_id}"
            )

        revenue_12m = (
            h12["SubTotal"].sum()
        )

        last_order = h12.iloc[-1]

        future_orders = orders[
            (
                orders["StoreID"]
                == store_id
            )
            & (
                orders["OrderDate"]
                >= snapshot
            )
            & (
                orders["OrderDate"]
                < label_end
            )
        ]

        row = {
            "StoreID": store_id,
            "snapshot": snapshot_text,

            "churn": int(
                future_orders.empty
            ),

            # Existing locked features
            "n_orders_3m":
                len(h3),

            "revenue_3m":
                h3["SubTotal"].sum(),

            "recency_days":
                (
                    snapshot
                    - h12[
                        "OrderDate"
                    ].max()
                ).days,

            "revenue_12m":
                revenue_12m,

            "share_bikes":
                h12[
                    "rev_bikes"
                ].sum()
                / revenue_12m,

            "share_accessories":
                h12[
                    "rev_accessories"
                ].sum()
                / revenue_12m,

            "share_clothing":
                h12[
                    "rev_clothing"
                ].sum()
                / revenue_12m,

            # New candidate features
            "avg_order_value_12m":
                h12[
                    "SubTotal"
                ].mean(),

            "std_order_value_12m":
                h12[
                    "SubTotal"
                ].std(ddof=1),

            "avg_qty_12m":
                h12[
                    "qty"
                ].mean(),

            "avg_lines_12m":
                h12[
                    "n_lines"
                ].mean(),

            "last_order_value":
                float(
                    last_order[
                        "SubTotal"
                    ]
                ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_rf_data(
    x_train,
    x_val,
):
    # std_order_value_12m is undefined
    # when there is only one historical order.
    #
    # Imputation is fitted on TRAIN only.

    imputer = SimpleImputer(
        strategy="median"
    )

    x_train = (
        imputer.fit_transform(
            x_train
        )
    )

    x_val = (
        imputer.transform(
            x_val
        )
    )

    return x_train, x_val


def predict_rf(
    x_train,
    y_train,
    x_val,
):
    x_train, x_val = (
        prepare_rf_data(
            x_train,
            x_val,
        )
    )

    probabilities = []

    for seed in SEEDS:

        model = (
            RandomForestClassifier(
                **RF_PARAMS,
                random_state=seed,
            )
        )

        model.fit(
            x_train,
            y_train,
        )

        probabilities.append(
            model.predict_proba(
                x_val
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def predict_gb(
    x_train,
    y_train,
    x_val,
):
    probabilities = []

    for seed in SEEDS:

        (
            x_balanced,
            y_balanced,
        ) = random_oversample(
            x_train,
            y_train,
            seed,
        )

        model = (
            HistGradientBoostingClassifier(
                **GB_PARAMS,
                random_state=seed,
            )
        )

        model.fit(
            x_balanced,
            y_balanced,
        )

        probabilities.append(
            model.predict_proba(
                x_val
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def safe_pr_auc(
    y_true,
    probabilities,
):
    y_true = np.asarray(
        y_true
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        < 2
    ):
        return np.nan

    return average_precision_score(
        y_true,
        probabilities,
    )


def evaluate_one(
    model_name,
    predictor,
    feature_name,
    frames,
):
    features = (
        BASE_FEATURES.copy()
    )

    if (
        feature_name
        != "BASELINE"
    ):
        features.append(
            feature_name
        )

    folds = [
        {
            "name": "fold_1",
            "train": [
                "2012-07-01",
            ],
            "validation":
                "2013-01-01",
        },
        {
            "name": "fold_2",
            "train": [
                "2012-07-01",
                "2012-10-01",
            ],
            "validation":
                "2013-04-01",
        },
    ]

    result = {
        "model":
            model_name,

        "feature_added":
            feature_name,
    }

    fold_scores = []

    for fold in folds:

        train = pd.concat(
            [
                frames[s]
                for s
                in fold["train"]
            ],
            ignore_index=True,
        )

        val = (
            frames[
                fold[
                    "validation"
                ]
            ].copy()
        )

        x_train = train[
            features
        ]

        y_train = train[
            "churn"
        ].astype(int)

        x_val = val[
            features
        ]

        y_val = val[
            "churn"
        ].astype(int)

        probabilities = (
            predictor(
                x_train,
                y_train,
                x_val,
            )
        )

        score = safe_pr_auc(
            y_val,
            probabilities,
        )

        fold_scores.append(
            score
        )

        result[
            f"{fold['name']}_pr_auc"
        ] = score

        # Special Fold-2 subgroup:
        # recently active resellers
        # who looked healthy immediately
        # before the snapshot.
        if (
            fold["name"]
            == "fold_2"
        ):
            recent_mask = (
                val[
                    "recency_days"
                ]
                <= 30
            )

            result[
                "fold_2_recent30_n"
            ] = int(
                recent_mask.sum()
            )

            result[
                "fold_2_recent30_churners"
            ] = int(
                val.loc[
                    recent_mask,
                    "churn",
                ].sum()
            )

            result[
                "fold_2_recent30_pr_auc"
            ] = safe_pr_auc(
                y_val.loc[
                    recent_mask
                ],
                probabilities[
                    recent_mask
                ],
            )

    result[
        "mean_pr_auc"
    ] = np.mean(
        fold_scores
    )

    result[
        "worst_fold_pr_auc"
    ] = np.min(
        fold_scores
    )

    return result


def print_snapshot_sanity(
    frames,
):
    print(
        "=== Snapshot sanity check ==="
    )

    expected = {
        "2012-07-01":
            (326, 76),

        "2012-10-01":
            (366, 47),

        "2013-01-01":
            (343, 25),

        "2013-04-01":
            (340, 64),
    }

    all_ok = True

    for (
        snapshot,
        frame,
    ) in frames.items():

        actual = (
            len(frame),
            int(
                frame[
                    "churn"
                ].sum()
            ),
        )

        wanted = expected[
            snapshot
        ]

        if (
            actual
            == wanted
        ):
            status = "PASS"
        else:
            status = "FAIL"
            all_ok = False

        print(
            f"{status} "
            f"{snapshot}: "
            f"rows={actual[0]}, "
            f"churners={actual[1]} "
            f"(expected "
            f"{wanted[0]}, "
            f"{wanted[1]})"
        )

    if not all_ok:
        raise RuntimeError(
            "Snapshot reconstruction "
            "does not match the "
            "verified development data."
        )

    print()


def add_deltas(
    results,
):
    results = results.copy()

    for model_name in (
        results[
            "model"
        ].unique()
    ):

        model_mask = (
            results[
                "model"
            ]
            == model_name
        )

        baseline = (
            results[
                model_mask
                & (
                    results[
                        "feature_added"
                    ]
                    == "BASELINE"
                )
            ]
            .iloc[0]
        )

        metrics = [
            "fold_1_pr_auc",
            "fold_2_pr_auc",
            "mean_pr_auc",
            "worst_fold_pr_auc",
            "fold_2_recent30_pr_auc",
        ]

        for metric in metrics:

            results.loc[
                model_mask,
                f"delta_{metric}",
            ] = (
                results.loc[
                    model_mask,
                    metric,
                ]
                - baseline[
                    metric
                ]
            )

    return results


def print_results(
    results,
):
    columns = [
        "model",
        "feature_added",

        "fold_1_pr_auc",
        "fold_2_pr_auc",

        "mean_pr_auc",
        "worst_fold_pr_auc",

        "delta_mean_pr_auc",
        "delta_fold_2_pr_auc",

        "fold_2_recent30_pr_auc",
        "delta_fold_2_recent30_pr_auc",
    ]

    print(
        "=== Targeted Add-One "
        "Feature Test ==="
    )

    print(
        results[
            columns
        ]
        .sort_values(
            [
                "model",
                "mean_pr_auc",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.4f}"
            ),
        )
    )

    print()

    print(
        "Fold-2 recent subgroup: "
        "recency_days <= 30 "
        "at 2013-04-01."
    )

    print()

    print(
        "Final-test snapshot "
        "2013-10-01 was NOT "
        "constructed or evaluated."
    )


def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        encoding="utf-8-sig",
    )

    orders[
        "OrderDate"
    ] = pd.to_datetime(
        orders[
            "OrderDate"
        ]
    )

    frames = {
        snapshot:
            build_snapshot(
                orders,
                snapshot,
            )
        for snapshot
        in SNAPSHOTS
    }

    print_snapshot_sanity(
        frames
    )

    experiments = [
        "BASELINE",
        *CANDIDATES,
    ]

    rows = []

    for feature_name in experiments:

        print(
            "Testing RF:",
            feature_name,
        )

        rows.append(
            evaluate_one(
                "Random Forest",
                predict_rf,
                feature_name,
                frames,
            )
        )

    for feature_name in experiments:

        print(
            "Testing GB:",
            feature_name,
        )

        rows.append(
            evaluate_one(
                "Gradient Boosting",
                predict_gb,
                feature_name,
                frames,
            )
        )

    results = (
        pd.DataFrame(
            rows
        )
    )

    results = add_deltas(
        results
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print_results(
        results
    )

    print()

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
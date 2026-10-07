from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)
ORDERS_PATH = Path(
    "datasets/orders_clean.csv"
)

RESULTS_DIR = Path(
    "results/feature_selection"
)
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "targeted_add_one_feature_results.csv"
)


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


FOLDS = {
    "fold_1": {
        "train": [
            "2012-07-01",
        ],
        "validation":
            "2013-01-01",
    },
    "fold_2": {
        "train": [
            "2012-07-01",
            "2012-10-01",
        ],
        "validation":
            "2013-04-01",
    },
}


SEEDS = [
    42,
    78,
    88,
    1034,
    2026,
]


RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


GB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}


# These are the saved Stage-08 results. The add-one
# experiment must reproduce them before any candidate
# comparison is trusted.
EXPECTED_BASELINE = {
    "Random Forest": {
        "fold_1": 0.409271934480239,
        "fold_2": 0.29468531757418653,
    },
    "Gradient Boosting": {
        "fold_1": 0.3480374636919924,
        "fold_2": 0.3057197779457027,
    },
}


def random_oversample(
    x,
    y,
    seed,
):
    x = x.reset_index(
        drop=True
    )

    y = y.reset_index(
        drop=True
    )

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
    ).reset_index(
        drop=True
    )

    y_balanced = balanced.pop(
        "_target"
    ).astype(int)

    return balanced, y_balanced


def build_rf(seed):
    # Exact Stage-08 Random Forest pipeline.
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
                    **RF_PARAMS,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_gb(seed):
    # Exact Stage-08 Gradient Boosting model.
    return HistGradientBoostingClassifier(
        **GB_PARAMS,
        random_state=seed,
        early_stopping=False,
    )


def build_candidate_features(
    orders,
    labeled,
):
    """
    Build only the five new candidate features.

    The existing seven baseline features are NOT rebuilt.
    They remain exactly as stored in ml_labeled_snapshots.csv.
    """
    key_rows = (
        labeled[
            [
                "StoreID",
                "snapshot",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    parts = []

    for snapshot_text, group in (
        key_rows.groupby(
            "snapshot",
            sort=True,
        )
    ):
        snapshot = pd.Timestamp(
            snapshot_text
        )

        start = (
            snapshot
            - pd.DateOffset(
                months=12
            )
        )

        store_ids = (
            group["StoreID"]
            .unique()
        )

        history = orders[
            (
                orders["StoreID"]
                .isin(store_ids)
            )
            & (
                orders["OrderDate"]
                >= start
            )
            & (
                orders["OrderDate"]
                < snapshot
            )
        ].copy()

        # Exactly one cleaned row exists per SalesOrderID,
        # but sort explicitly so the last order is deterministic
        # if multiple resellers order on the same date.
        history = history.sort_values(
            [
                "StoreID",
                "OrderDate",
                "SalesOrderID",
            ]
        )

        aggregated = (
            history.groupby(
                "StoreID",
                as_index=False,
            )
            .agg(
                avg_order_value_12m=(
                    "SubTotal",
                    "mean",
                ),
                std_order_value_12m=(
                    "SubTotal",
                    "std",
                ),
                avg_qty_12m=(
                    "qty",
                    "mean",
                ),
                avg_lines_12m=(
                    "n_lines",
                    "mean",
                ),
            )
        )

        last_orders = (
            history.groupby(
                "StoreID",
                as_index=False,
            )
            .tail(1)[
                [
                    "StoreID",
                    "SubTotal",
                ]
            ]
            .rename(
                columns={
                    "SubTotal":
                        "last_order_value"
                }
            )
        )

        features = (
            group[
                [
                    "StoreID",
                    "snapshot",
                ]
            ]
            .merge(
                aggregated,
                on="StoreID",
                how="left",
                validate="one_to_one",
            )
            .merge(
                last_orders,
                on="StoreID",
                how="left",
                validate="one_to_one",
            )
        )

        parts.append(
            features
        )

    candidates = pd.concat(
        parts,
        ignore_index=True,
    )

    assert len(
        candidates
    ) == len(
        key_rows
    )

    assert not candidates.duplicated(
        [
            "StoreID",
            "snapshot",
        ]
    ).any()

    # Every labeled reseller is eligible because it bought
    # within the previous six months, therefore it must have
    # at least one order in the previous twelve months.
    required = [
        "avg_order_value_12m",
        "avg_qty_12m",
        "avg_lines_12m",
        "last_order_value",
    ]

    assert (
        candidates[
            required
        ]
        .notna()
        .all()
        .all()
    )

    # std_order_value_12m is intentionally NaN for a reseller
    # with only one order in the 12-month history.
    return candidates


def attach_candidate_features(
    labeled,
    candidates,
):
    original_rows = len(
        labeled
    )

    original_keys = (
        labeled[
            [
                "StoreID",
                "snapshot",
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    result = labeled.merge(
        candidates,
        on=[
            "StoreID",
            "snapshot",
        ],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    assert len(
        result
    ) == original_rows

    pd.testing.assert_frame_equal(
        result[
            [
                "StoreID",
                "snapshot",
            ]
        ].reset_index(
            drop=True
        ),
        original_keys,
        check_dtype=False,
    )

    return result


def evaluate_rf_seed(
    x_train,
    y_train,
    x_val,
    y_val,
    recent_mask,
    seed,
):
    model = build_rf(
        seed
    )

    model.fit(
        x_train,
        y_train,
    )

    probability = (
        model.predict_proba(
            x_val
        )[:, 1]
    )

    overall = (
        average_precision_score(
            y_val,
            probability,
        )
    )

    recent = (
        average_precision_score(
            y_val.loc[
                recent_mask
            ],
            probability[
                recent_mask.to_numpy()
            ],
        )
    )

    return overall, recent


def evaluate_gb_seed(
    x_train,
    y_train,
    x_val,
    y_val,
    recent_mask,
    seed,
):
    x_balanced, y_balanced = (
        random_oversample(
            x_train,
            y_train,
            seed,
        )
    )

    model = build_gb(
        seed
    )

    model.fit(
        x_balanced,
        y_balanced,
    )

    probability = (
        model.predict_proba(
            x_val
        )[:, 1]
    )

    overall = (
        average_precision_score(
            y_val,
            probability,
        )
    )

    recent = (
        average_precision_score(
            y_val.loc[
                recent_mask
            ],
            probability[
                recent_mask.to_numpy()
            ],
        )
    )

    return overall, recent


def evaluate_experiment(
    df,
    model_name,
    feature_added,
):
    features = (
        BASE_FEATURES.copy()
    )

    if (
        feature_added
        != "BASELINE"
    ):
        features.append(
            feature_added
        )

    fold_scores = {}
    recent_scores = {}

    recent_n = None
    recent_churners = None

    for (
        fold_name,
        fold,
    ) in FOLDS.items():

        train = df[
            df["snapshot"].isin(
                fold["train"]
            )
        ].copy()

        val = df[
            df["snapshot"]
            == fold["validation"]
        ].copy()

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

        recent_mask = (
            val["recency_days"]
            <= 30
        )

        seed_scores = []
        seed_recent_scores = []

        for seed in SEEDS:

            if (
                model_name
                == "Random Forest"
            ):
                overall, recent = (
                    evaluate_rf_seed(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        recent_mask,
                        seed,
                    )
                )

            elif (
                model_name
                == "Gradient Boosting"
            ):
                overall, recent = (
                    evaluate_gb_seed(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        recent_mask,
                        seed,
                    )
                )

            else:
                raise ValueError(
                    model_name
                )

            seed_scores.append(
                overall
            )

            seed_recent_scores.append(
                recent
            )

        # IMPORTANT:
        # This matches Stage 08 exactly:
        # score each seed first, then average scores.
        fold_scores[
            fold_name
        ] = np.mean(
            seed_scores
        )

        recent_scores[
            fold_name
        ] = np.mean(
            seed_recent_scores
        )

        if (
            fold_name
            == "fold_2"
        ):
            recent_n = int(
                recent_mask.sum()
            )

            recent_churners = int(
                y_val.loc[
                    recent_mask
                ].sum()
            )

    return {
        "model":
            model_name,

        "feature_added":
            feature_added,

        "fold_1_pr_auc":
            fold_scores[
                "fold_1"
            ],

        "fold_2_pr_auc":
            fold_scores[
                "fold_2"
            ],

        "mean_pr_auc":
            np.mean(
                list(
                    fold_scores.values()
                )
            ),

        "worst_fold_pr_auc":
            np.min(
                list(
                    fold_scores.values()
                )
            ),

        "fold_2_recent30_n":
            recent_n,

        "fold_2_recent30_churners":
            recent_churners,

        "fold_2_recent30_pr_auc":
            recent_scores[
                "fold_2"
            ],
    }


def verify_baseline(
    results,
):
    print()
    print(
        "=== Baseline Reproduction Check ==="
    )

    for (
        model_name,
        expected,
    ) in EXPECTED_BASELINE.items():

        row = results[
            (
                results["model"]
                == model_name
            )
            & (
                results["feature_added"]
                == "BASELINE"
            )
        ].iloc[0]

        for fold_name in [
            "fold_1",
            "fold_2",
        ]:
            actual = row[
                f"{fold_name}_pr_auc"
            ]

            target = expected[
                fold_name
            ]

            difference = (
                actual
                - target
            )

            print(
                f"{model_name} "
                f"{fold_name}: "
                f"actual={actual:.10f}, "
                f"expected={target:.10f}, "
                f"delta={difference:+.10f}"
            )

            if not np.isclose(
                actual,
                target,
                rtol=0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    "Baseline reproduction failed. "
                    "Do not interpret candidate "
                    "feature results."
                )

    print(
        "[PASS] Exact Stage-08 "
        "baseline reproduced."
    )


def add_deltas(
    results,
):
    results = results.copy()

    metrics = [
        "fold_1_pr_auc",
        "fold_2_pr_auc",
        "mean_pr_auc",
        "worst_fold_pr_auc",
        "fold_2_recent30_pr_auc",
    ]

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

    print()
    print(
        "=== Corrected Targeted "
        "Add-One Feature Test ==="
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
        "Existing seven features came directly "
        "from ml_labeled_snapshots.csv."
    )

    print(
        "Final-test snapshot 2013-10-01 "
        "was NOT constructed or evaluated."
    )


def main():
    labeled = pd.read_csv(
        DATA_PATH
    )

    labeled[
        "snapshot"
    ] = pd.to_datetime(
        labeled[
            "snapshot"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=[
            "OrderDate",
        ],
        encoding="utf-8-sig",
    )

    expected_counts = {
        "2012-07-01":
            (326, 76),

        "2012-10-01":
            (366, 47),

        "2013-01-01":
            (343, 25),

        "2013-04-01":
            (340, 64),
    }

    print(
        "=== Development Dataset Check ==="
    )

    for (
        snapshot,
        expected,
    ) in expected_counts.items():

        part = labeled[
            labeled["snapshot"]
            == snapshot
        ]

        actual = (
            len(part),
            int(
                part[
                    "churn"
                ].sum()
            ),
        )

        assert actual == expected

        print(
            f"[PASS] {snapshot}: "
            f"{actual[0]} rows, "
            f"{actual[1]} churners"
        )

    candidates = (
        build_candidate_features(
            orders,
            labeled,
        )
    )

    df = attach_candidate_features(
        labeled,
        candidates,
    )

    experiments = [
        "BASELINE",
        *CANDIDATES,
    ]

    rows = []

    for model_name in [
        "Random Forest",
        "Gradient Boosting",
    ]:

        for feature_added in experiments:

            print(
                f"Testing "
                f"{model_name}: "
                f"{feature_added}"
            )

            rows.append(
                evaluate_experiment(
                    df,
                    model_name,
                    feature_added,
                )
            )

    results = pd.DataFrame(
        rows
    )

    # Critical guardrail: if this fails,
    # no candidate result should be trusted.
    verify_baseline(
        results
    )

    results = add_deltas(
        results
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

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

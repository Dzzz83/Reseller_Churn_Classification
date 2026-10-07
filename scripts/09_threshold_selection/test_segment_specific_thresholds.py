from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path(
    "datasets/processed/ml_labeled_snapshots.csv"
)

RESULTS_DIR = Path("results/threshold_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    RESULTS_DIR
    / "segment_threshold_fold1_to_fold2_summary.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "segment_threshold_fold2_predictions.csv"
)


FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]


FOLD_1 = {
    "train": ["2012-07-01"],
    "validation": "2013-01-01",
}

FOLD_2 = {
    "train": [
        "2012-07-01",
        "2012-10-01",
    ],
    "validation": "2013-04-01",
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


THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01,
)


# Fixed from the diagnostic definition.
RECENT_CUTOFF_DAYS = 30


def random_oversample(
    x,
    y,
    seed,
):
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


def build_rf(seed):
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
    return HistGradientBoostingClassifier(
        **GB_PARAMS,
        random_state=seed,
        early_stopping=False,
    )


def prepare_fold(
    df,
    fold,
):
    train = df[
        df["snapshot"].isin(
            fold["train"]
        )
    ].copy()

    val = df[
        df["snapshot"]
        == fold["validation"]
    ].copy()

    return {
        "train":
            train,

        "val":
            val,

        "x_train":
            train[FEATURES],

        "y_train":
            train["churn"].astype(int),

        "x_val":
            val[FEATURES],

        "y_val":
            val["churn"].astype(int),
    }


def get_rf_probabilities(
    data,
):
    probabilities = []

    for seed in SEEDS:
        model = build_rf(seed)

        model.fit(
            data["x_train"],
            data["y_train"],
        )

        probabilities.append(
            model.predict_proba(
                data["x_val"]
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def get_gb_probabilities(
    data,
):
    probabilities = []

    for seed in SEEDS:
        (
            x_balanced,
            y_balanced,
        ) = random_oversample(
            data["x_train"],
            data["y_train"],
            seed,
        )

        model = build_gb(seed)

        model.fit(
            x_balanced,
            y_balanced,
        )

        probabilities.append(
            model.predict_proba(
                data["x_val"]
            )[:, 1]
        )

    return np.mean(
        probabilities,
        axis=0,
    )


def metric_row(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    return {
        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def choose_best_threshold(
    y_true,
    probabilities,
):
    """
    Choose a threshold on one development segment.

    Primary criterion:
      1. highest F1

    Tie breakers:
      2. higher recall
      3. higher precision
      4. higher threshold

    The final tie breaker favors the more conservative
    threshold when all classification metrics are equal.
    """
    rows = []

    for threshold in THRESHOLDS:
        y_pred = (
            probabilities
            >= threshold
        ).astype(int)

        metrics = metric_row(
            y_true,
            y_pred,
        )

        rows.append(
            {
                "threshold":
                    threshold,

                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        [
            "f1",
            "recall",
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return result.iloc[0]


def select_fold1_thresholds(
    val,
    y_true,
    probabilities,
):
    recent_mask = (
        val["recency_days"]
        <= RECENT_CUTOFF_DAYS
    ).to_numpy()

    other_mask = ~recent_mask

    global_best = (
        choose_best_threshold(
            y_true,
            probabilities,
        )
    )

    recent_best = (
        choose_best_threshold(
            y_true[
                recent_mask
            ],
            probabilities[
                recent_mask
            ],
        )
    )

    other_best = (
        choose_best_threshold(
            y_true[
                other_mask
            ],
            probabilities[
                other_mask
            ],
        )
    )

    return {
        "global":
            global_best,

        "recent":
            recent_best,

        "other":
            other_best,
    }


def apply_global_threshold(
    probabilities,
    threshold,
):
    return (
        probabilities
        >= threshold
    ).astype(int)


def apply_segment_thresholds(
    val,
    probabilities,
    recent_threshold,
    other_threshold,
):
    recent_mask = (
        val["recency_days"]
        <= RECENT_CUTOFF_DAYS
    ).to_numpy()

    thresholds = np.where(
        recent_mask,
        recent_threshold,
        other_threshold,
    )

    return (
        probabilities
        >= thresholds
    ).astype(int)


def evaluate_fold2_strategy(
    model_name,
    strategy_name,
    val,
    y_true,
    y_pred,
):
    rows = []

    masks = {
        "overall":
            np.ones(
                len(val),
                dtype=bool,
            ),

        "recent":
            (
                val["recency_days"]
                <= RECENT_CUTOFF_DAYS
            ).to_numpy(),

        "other":
            (
                val["recency_days"]
                > RECENT_CUTOFF_DAYS
            ).to_numpy(),
    }

    for (
        group_name,
        mask,
    ) in masks.items():

        metrics = metric_row(
            y_true[mask],
            y_pred[mask],
        )

        rows.append(
            {
                "model":
                    model_name,

                "strategy":
                    strategy_name,

                "group":
                    group_name,

                "n":
                    int(
                        mask.sum()
                    ),

                "actual_churners":
                    int(
                        y_true[
                            mask
                        ].sum()
                    ),

                **metrics,
            }
        )

    return rows


def print_fold1_selection(
    model_name,
    selected,
):
    print()
    print(
        f"=== {model_name}: "
        "Fold-1 Threshold Selection ==="
    )

    for name in [
        "global",
        "recent",
        "other",
    ]:
        row = selected[
            name
        ]

        print(
            f"{name:>7}: "
            f"threshold="
            f"{row['threshold']:.2f}, "
            f"precision="
            f"{row['precision']:.2%}, "
            f"recall="
            f"{row['recall']:.2%}, "
            f"F1="
            f"{row['f1']:.4f}"
        )


def main():
    df = pd.read_csv(
        DATA_PATH
    )

    df[
        "snapshot"
    ] = pd.to_datetime(
        df[
            "snapshot"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    fold1 = prepare_fold(
        df,
        FOLD_1,
    )

    fold2 = prepare_fold(
        df,
        FOLD_2,
    )

    print(
        "=== Segment-Specific "
        "Threshold Experiment ==="
    )

    print(
        "Threshold selection: "
        "Fold 1 ONLY"
    )

    print(
        "Evaluation: "
        "Fold 2"
    )

    print(
        f"Recent segment: "
        f"recency_days <= "
        f"{RECENT_CUTOFF_DAYS}"
    )

    print(
        "Final-test labels: "
        "NOT USED"
    )

    results = []
    prediction_exports = (
        fold2["val"][
            [
                "StoreID",
                "snapshot",
                "churn",
                "recency_days",
            ]
        ].copy()
    )

    model_specs = [
        (
            "Random Forest",
            get_rf_probabilities,
        ),
        (
            "Gradient Boosting",
            get_gb_probabilities,
        ),
    ]

    for (
        model_name,
        probability_function,
    ) in model_specs:

        print()
        print(
            f"Training "
            f"{model_name}..."
        )

        fold1_probabilities = (
            probability_function(
                fold1
            )
        )

        fold2_probabilities = (
            probability_function(
                fold2
            )
        )

        fold1_y = (
            fold1[
                "y_val"
            ].to_numpy()
        )

        fold2_y = (
            fold2[
                "y_val"
            ].to_numpy()
        )

        selected = (
            select_fold1_thresholds(
                fold1["val"],
                fold1_y,
                fold1_probabilities,
            )
        )

        print_fold1_selection(
            model_name,
            selected,
        )

        global_threshold = (
            selected[
                "global"
            ][
                "threshold"
            ]
        )

        recent_threshold = (
            selected[
                "recent"
            ][
                "threshold"
            ]
        )

        other_threshold = (
            selected[
                "other"
            ][
                "threshold"
            ]
        )

        global_pred = (
            apply_global_threshold(
                fold2_probabilities,
                global_threshold,
            )
        )

        segment_pred = (
            apply_segment_thresholds(
                fold2["val"],
                fold2_probabilities,
                recent_threshold,
                other_threshold,
            )
        )

        results.extend(
            evaluate_fold2_strategy(
                model_name,
                "fold1_global_threshold",
                fold2["val"],
                fold2_y,
                global_pred,
            )
        )

        results.extend(
            evaluate_fold2_strategy(
                model_name,
                "fold1_segment_thresholds",
                fold2["val"],
                fold2_y,
                segment_pred,
            )
        )

        safe_name = (
            "rf"
            if model_name
            == "Random Forest"
            else "gb"
        )

        prediction_exports[
            f"{safe_name}_probability"
        ] = fold2_probabilities

        prediction_exports[
            f"{safe_name}_global_pred"
        ] = global_pred

        prediction_exports[
            f"{safe_name}_segment_pred"
        ] = segment_pred

        prediction_exports[
            f"{safe_name}_applied_segment_threshold"
        ] = np.where(
            (
                fold2["val"][
                    "recency_days"
                ]
                <= RECENT_CUTOFF_DAYS
            ).to_numpy(),
            recent_threshold,
            other_threshold,
        )

        # Store chosen thresholds on every summary row
        # for this model so the CSV is self-contained.
        for row in results:
            if (
                row["model"]
                == model_name
            ):
                row[
                    "fold1_global_threshold"
                ] = global_threshold

                row[
                    "fold1_recent_threshold"
                ] = recent_threshold

                row[
                    "fold1_other_threshold"
                ] = other_threshold

    summary = pd.DataFrame(
        results
    )

    print()
    print(
        "=== Fold-2 Comparison ==="
    )

    display_columns = [
        "model",
        "strategy",
        "group",
        "actual_churners",
        "tp",
        "fn",
        "fp",
        "precision",
        "recall",
        "f1",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=== Overall Fold-2 "
        "Strategy Change ==="
    )

    for model_name in (
        summary[
            "model"
        ].unique()
    ):
        base = summary[
            (
                summary[
                    "model"
                ]
                == model_name
            )
            & (
                summary[
                    "strategy"
                ]
                == "fold1_global_threshold"
            )
            & (
                summary[
                    "group"
                ]
                == "overall"
            )
        ].iloc[0]

        segmented = summary[
            (
                summary[
                    "model"
                ]
                == model_name
            )
            & (
                summary[
                    "strategy"
                ]
                == "fold1_segment_thresholds"
            )
            & (
                summary[
                    "group"
                ]
                == "overall"
            )
        ].iloc[0]

        recent_base = summary[
            (
                summary[
                    "model"
                ]
                == model_name
            )
            & (
                summary[
                    "strategy"
                ]
                == "fold1_global_threshold"
            )
            & (
                summary[
                    "group"
                ]
                == "recent"
            )
        ].iloc[0]

        recent_segmented = summary[
            (
                summary[
                    "model"
                ]
                == model_name
            )
            & (
                summary[
                    "strategy"
                ]
                == "fold1_segment_thresholds"
            )
            & (
                summary[
                    "group"
                ]
                == "recent"
            )
        ].iloc[0]

        print()
        print(model_name)

        print(
            "  Overall recall: "
            f"{base['recall']:.2%} "
            "-> "
            f"{segmented['recall']:.2%}"
        )

        print(
            "  Overall precision: "
            f"{base['precision']:.2%} "
            "-> "
            f"{segmented['precision']:.2%}"
        )

        print(
            "  Overall F1: "
            f"{base['f1']:.4f} "
            "-> "
            f"{segmented['f1']:.4f}"
        )

        print(
            "  Recent churn recall: "
            f"{recent_base['recall']:.2%} "
            "-> "
            f"{recent_segmented['recall']:.2%}"
        )

        print(
            "  False alarms: "
            f"{int(base['fp'])} "
            "-> "
            f"{int(segmented['fp'])}"
        )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    prediction_exports.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        "Saved:",
        SUMMARY_PATH,
    )

    print(
        "Saved:",
        PREDICTIONS_PATH,
    )

    print()
    print(
        "Interpretation rule:"
    )

    print(
        "- Fold 1 chooses all thresholds."
    )

    print(
        "- Fold 2 only evaluates the "
        "already-frozen thresholds."
    )

    print(
        "- The 30-day segmentation rule "
        "is still a development hypothesis."
    )

    print(
        "- Final-test labels were NOT used."
    )


if __name__ == "__main__":
    main()

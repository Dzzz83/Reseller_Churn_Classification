"""Self-contained development cases; never load repository datasets or results."""

import pandas as pd
import pytest

from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS
from reseller_churn.modeling.candidates import Candidate


@pytest.fixture
def grouped_data():
    records = []
    for snapshot in TRAINING_SNAPSHOTS:
        for store in range(1, 33):
            count = store % 3 + 1
            records.append(
                {
                    "StoreID": store,
                    "snapshot": pd.Timestamp(snapshot),
                    "churn": int(store % 4 == 0),
                    "n_orders_3m": count,
                    "revenue_3m": count * 10.0,
                    "recency_days": store % 30 + 1,
                    "share_bikes": 0.7,
                    "share_accessories": 0.1,
                    "share_clothing": 0.1,
                    "revenue_12m": count * 40.0,
                    "weighted_discount_3m": store / 100,
                    "discount_change_3m": 0.0,
                    "discounted_line_share_12m": 1.0,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def small_candidates():
    return [
        Candidate(
            architecture="hist_gradient_boosting",
            feature_set="ten",
            imbalance_strategy=strategy,
            parameters={
                "learning_rate": 0.1,
                "max_iter": 3,
                "max_leaf_nodes": 3,
                "min_samples_leaf": 2,
                "l2_regularization": 0.0,
            },
            seeds=(42,),
        )
        for strategy in ("balanced_weights", "period_class_weights")
    ]

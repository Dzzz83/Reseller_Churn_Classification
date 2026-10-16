"""Configuration frozen before the original October 2013 outcome access."""

FEATURES = (
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
    "weighted_discount_3m",
    "discount_change_3m",
    "discounted_line_share_12m",
)
DISCOUNT_FEATURES = FEATURES[-3:]
SEEDS = (42, 78, 88, 1034, 2026)
PARAMETERS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
}
TRAINING_SNAPSHOTS = ("2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01")
ORIGINAL_TEST_SNAPSHOT = "2013-10-01"
PRIMARY_CUTOFF = 0.06
ORIGINAL_RULES = {
    0.01: "secondary",
    0.06: "primary",
    0.55: "secondary",
    0.88: "secondary",
}


def model_configuration() -> dict:
    return {
        "architecture": "hist_gradient_boosting",
        "features": list(FEATURES),
        "parameters": dict(PARAMETERS),
        "seeds": list(SEEDS),
        "seed_aggregation": "mean_predicted_probability",
        "imbalance_strategy": "balanced_training_weights",
        "missing_handling": "native",
        "training_snapshots": list(TRAINING_SNAPSHOTS),
        "original_primary_cutoff": PRIMARY_CUTOFF,
    }

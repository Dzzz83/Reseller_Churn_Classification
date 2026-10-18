"""Fixed production feature order, estimator settings, seeds and retained cutoff."""

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
    "discount_frequency_12m",
)
# The retained binary predates the public rename. Adapt names only at inference;
# never rewrite the binary or change the feature order/values.
RETAINED_FEATURES = (*FEATURES[:-1], "discounted_line_share_12m")
PURCHASE_FEATURES = FEATURES[:7]
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
CUTOFF = 0.27912545951329487
MODEL_ID = "period_class_weighted"

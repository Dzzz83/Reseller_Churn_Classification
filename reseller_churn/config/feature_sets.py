"""Named feature sets used by experiments."""

RFM_FEATURES = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
]

FULL_FEATURES = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
    "n_orders_6m",
    "n_orders_3m",
    "revenue_6m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
    "overdue_ratio",
    "has_previous_6m_revenue",
    "revenue_trend",
    "share_bikes",
    "share_components",
    "share_clothing",
    "share_accessories",
    "store_age",
]

REDUCED_FEATURES = [
    "recency_days",
    "n_orders_12m",
    "revenue_12m",
    "n_orders_6m",
    "n_orders_3m",
    "revenue_3m",
    "mean_gap",
    "std_gap",
    "has_previous_6m_revenue",
    "revenue_trend",
    "share_bikes",
    "share_clothing",
    "share_accessories",
    "store_age",
]

PRUNED_FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
]

FEATURE_SETS = {
    "rfm": RFM_FEATURES,
    "full": FULL_FEATURES,
    "reduced": REDUCED_FEATURES,
    "pruned": PRUNED_FEATURES,
}

"""Weights derived exclusively from the rows in a fitting partition."""

import numpy as np

from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS


def period_weights(train):
    """Give each of the four period × two outcome cells equal total influence."""
    if set(train.snapshot.dt.strftime("%Y-%m-%d")) != set(TRAINING_SNAPSHOTS):
        raise ValueError("All four approved fitting periods required")
    counts = train.groupby(["snapshot", "churn"]).size()
    if len(counts) != 8 or not train.churn.isin([0, 1]).all():
        raise ValueError("Both fitting outcomes required in every period")
    return np.array(
        [
            len(train) / (8 * counts.loc[(row.snapshot, row.churn)])
            for row in train[["snapshot", "churn"]].itertuples(index=False)
        ]
    )

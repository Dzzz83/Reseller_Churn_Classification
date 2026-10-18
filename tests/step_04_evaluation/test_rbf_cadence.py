"""SVM scoring is reconstructible and excludes internal ungrouped calibration."""

import numpy as np
import pandas as pd
from scipy.special import expit

from comparisons.crossfit_percentile import FEATURES
from comparisons.rbf_cadence import fit_margin
from comparisons.verify_rbf_cadence import design, margins


def test_synthetic_rbf_smoke_missing_constant_extrapolation_and_kernel_reconstruction():
    rng = np.random.default_rng(91)
    train = pd.DataFrame(rng.normal(size=(160, len(FEATURES))), columns=FEATURES)
    train["StoreID"] = np.arange(1, 161)
    train["snapshot"] = pd.Timestamp("2013-04-01")
    train["churn"] = (train.recency_days * train.revenue_3m > 0.2).astype(int)
    train["n_orders_3m"] = 1.0
    train["weighted_discount_3m"] = np.nan
    train.loc[::4, "gap_cv"] = np.nan
    held = train.iloc[:8][FEATURES].copy()
    held.iloc[0, 1] = 1e6
    held.iloc[2] = held.iloc[1]
    score, records, spread, _ = fit_margin(train, held, {"id": "synthetic"})
    identity = records[0]
    assert len(records) == 1 and spread == 0 and np.isfinite(score).all()
    assert not identity["parameters"]["probability"] and identity["fit_status"] == 0
    np.testing.assert_allclose(score, expit(margins(held, identity)), atol=2e-13)
    assert score[1] == score[2]
    np.testing.assert_array_equal(
        design(train, identity)[identity["support_indices"]],
        identity["support_vectors"],
    )
    bounds = np.array(identity["weights"])[identity["support_indices"]]
    assert np.all(np.abs(identity["dual_coefficients"]) <= bounds + 1e-9)
    assert FEATURES.index("weighted_discount_3m") in identity["indicator_columns"]

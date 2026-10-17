"""Linear preprocessing statistics must be fitted on training rows only."""

import numpy as np

from comparisons.grouped_precision_recall import FEATURES, synthetic_development
from comparisons.linear_comparison import fit_linear


def test_linear_preprocessing_excludes_validation_and_converges():
    data = synthetic_development()
    train = data.iloc[:200].copy()
    train.loc[train.index[:30], "discount_change_3m"] = np.nan
    valid = data.iloc[200:205][FEATURES].copy()
    valid["revenue_3m"] = 1e10
    score, identity = fit_linear(train, valid)
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    medians = train[FEATURES].median().to_numpy()
    np.testing.assert_allclose(identity["imputer_statistics"], medians, rtol=0, atol=0)
    missing = train[FEATURES].isna().to_numpy()
    columns = np.flatnonzero(missing.any(axis=0))
    expected = np.column_stack(
        [np.where(missing, medians, train[FEATURES]), missing[:, columns]]
    )
    np.testing.assert_allclose(
        identity["scaler_mean"], expected.mean(axis=0), rtol=1e-12, atol=1e-12
    )
    assert identity["iterations"] < 2000
    assert (
        len(identity["coefficient_names"])
        == len(identity["coefficients"])
        == expected.shape[1]
    )

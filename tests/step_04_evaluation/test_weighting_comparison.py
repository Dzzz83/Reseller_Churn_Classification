"""Unit-weight fits must not silently restore class/sample weighting."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from comparisons.grouped_precision_recall import FEATURES, synthetic_development
from comparisons.weighting_comparison import fit_unweighted
from reseller_churn.config.model_settings import RandomForestSettings


@pytest.mark.parametrize("method", ["unweighted_hgb", "unweighted_forest"])
def test_unit_weight_fit_omits_sample_weight(method, monkeypatch):
    estimator_class = (
        HistGradientBoostingClassifier if method == "unweighted_hgb" else Pipeline
    )
    original = estimator_class.fit
    seen = []

    def capture(self, x, y, **kwargs):
        seen.append(kwargs.copy())
        assert not any("sample_weight" in key for key in kwargs)
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(estimator_class, "fit", capture)
    data = synthetic_development()
    score, identities = fit_unweighted(
        method,
        data.iloc[:200],
        data.iloc[200:205][FEATURES],
        seeds=(42,),
        forest_settings=RandomForestSettings(n_estimators=8),
    )
    assert len(seen) == 1 and len(score) == 5
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    assert identities[0]["training_weighting"] == "unit/no sample_weight"
    assert identities[0]["parameters"]["class_weight"] is None

"""Retained learner smoke and historical scoring boundaries for the fixed sweep."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from comparisons import grouped_precision_recall as study
from comparisons.all_candidate_precision import fit_spec, historical_share, score_models
from comparisons.crossfit_percentile import FEATURES


@pytest.mark.parametrize(
    "name",
    [
        "smooth_cadence",
        "reseller_weighted",
        "activity_experts",
        "period_class_weighted",
        "rbf_cadence",
        "early_quarter_share",
    ],
)
def test_retained_synthetic_learner_reproduces_scores_after_serialization(name):
    data = study.synthetic_development()
    data["early_quarter_order_share"] = 0.25
    train = data[data.StoreID <= 60]
    held = data[data.StoreID > 60].iloc[:12]
    columns = FEATURES if name in ["smooth_cadence", "rbf_cadence"] else study.FEATURES
    if name == "early_quarter_share":
        columns = columns + ["early_quarter_order_share"]
    spec = dict(
        id=name, features=columns, parameters={**study.BASE_PARAMETERS, "max_iter": 10}
    )
    features = held[study.KEYS + columns]
    scores, identities, _, models = fit_spec(train, features, spec, retain=True)
    assert models and identities and np.isfinite(scores).all()
    stream = io.BytesIO()
    joblib.dump(dict(spec=spec, models=models), stream)
    stream.seek(0)
    saved = joblib.load(stream)
    np.testing.assert_allclose(
        scores, score_models(saved, features), rtol=0, atol=1e-15
    )
    with pytest.raises(ValueError, match="Held outcomes"):
        fit_spec(train, features.assign(churn=0), spec)


def test_historical_share_excludes_future_and_preserves_quarter_boundaries():
    keys = pd.DataFrame(dict(StoreID=[1], snapshot=pd.to_datetime(["2013-10-01"])))
    orders = pd.DataFrame(
        dict(
            StoreID=[1] * 5,
            SalesOrderID=range(1, 6),
            OrderDate=pd.to_datetime(
                ["2012-09-30", "2012-10-01", "2013-01-01", "2013-09-30", "2013-10-01"]
            ),
        )
    )
    result = historical_share(keys, orders)
    assert result.annual_orders.iloc[0] == 3
    assert result.early_quarter_order_share.iloc[0] == 1 / 3
    with pytest.raises(ValueError, match="Duplicate"):
        historical_share(keys, pd.concat([orders, orders.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="Unapproved"):
        historical_share(keys.assign(snapshot=pd.Timestamp("2014-01-01")), orders)


def test_frozen_inventory_covers_every_predeclared_candidate():
    from comparisons.all_candidate_precision import NAMES, STUDIES

    cached = {
        "baseline",
        "retained_logistic",
        "positive_floor_veto",
        "cadence_unit_revenue",
    }
    derived = {"complementary_mean", "guarded_component"}
    assert len(NAMES) == 13 and len(set(NAMES)) == 13
    assert set(NAMES) == cached | derived | set(STUDIES)

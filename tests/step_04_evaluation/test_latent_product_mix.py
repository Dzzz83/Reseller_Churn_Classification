"""Purchase-mix projection is fit on training histories, without outcome access."""

import numpy as np
import pandas as pd

from comparisons.latent_product_mix import (
    BASE_FEATURES,
    fit_latent_mix,
    learn_projection,
    project_cases,
)


def source():
    keys = pd.DataFrame({"StoreID": range(20), "snapshot": pd.Timestamp("2013-04-01")})
    values = np.random.default_rng(42).dirichlet(np.ones(6), size=20)
    rows = []
    for scope in ("annual", "latest_order"):
        for store in range(20):
            for index, share in enumerate(values[store]):
                rows.append(
                    dict(
                        StoreID=store,
                        snapshot=pd.Timestamp("2013-04-01"),
                        ProductID=index + 10,
                        scope=scope,
                        share=share,
                    )
                )
    return keys, pd.DataFrame(rows)


def test_projection_is_training_only_and_preserves_unseen_mass():
    keys, profiles = source()
    train, valid = keys.iloc[:12], keys.iloc[12:]
    projection = learn_projection(train, profiles)
    changed = profiles.copy()
    changed.loc[changed.StoreID.ge(12), "ProductID"] = 99
    changed = changed.groupby(
        ["StoreID", "snapshot", "ProductID", "scope"], as_index=False
    ).share.sum()
    assert learn_projection(train, changed) == projection
    assert projection["vocabulary"] == list(range(10, 16))
    components = np.array(projection["components"])
    np.testing.assert_allclose(components @ components.T, np.eye(5), atol=1e-12)
    expanded, columns = project_cases(train, profiles, projection)
    np.testing.assert_allclose(expanded[columns[:5]].mean(), 0, atol=1e-12)
    unseen, _ = project_cases(valid, changed, projection)
    np.testing.assert_allclose(unseen.annual_unseen_share, 1)
    assert np.isfinite(unseen[columns]).all().all()


def test_projected_hgb_smoke_has_training_identities_and_no_validation_labels():
    keys, profiles = source()
    cases = keys.assign(**{feature: 1.0 for feature in BASE_FEATURES})
    train = cases.iloc[:12].assign(churn=[0, 1] * 6)
    score, identities = fit_latent_mix(train, cases.iloc[12:], profiles)
    assert len(score) == 8 and np.isfinite(score).all()
    assert len(identities) == 5
    assert all(len(row["features"]) == 24 for row in identities)
    assert all(
        row["projection"]["vocabulary"] == list(range(10, 16)) for row in identities
    )


def test_frozen_component_count_fails_explicitly_when_insufficient():
    import pytest

    keys, profiles = source()
    with pytest.raises(ValueError, match="Insufficient"):
        learn_projection(keys, profiles[profiles.ProductID.lt(13)])

"""Historical SKU profiles obey the prediction window and source contract."""

import numpy as np
import pandas as pd
import pytest

from comparisons.product_mix import historical_product_shares, matched_sku_tables


def source():
    keys = pd.DataFrame({"StoreID": [1], "snapshot": pd.to_datetime(["2013-04-01"])})
    orders = pd.DataFrame(
        {
            "StoreID": [1] * 4,
            "SalesOrderID": [1, 2, 3, 4],
            "OrderDate": pd.to_datetime(
                ["2012-03-31", "2012-04-01", "2013-03-31", "2013-04-01"]
            ),
            "SubTotal": [10.0, 30.0, 70.0, 100.0],
        }
    )
    lines = pd.DataFrame(
        {
            "SalesOrderID": [1, 2, 3, 3, 4],
            "SalesOrderDetailID": range(5),
            "ProductID": [99, 7, 7, 8, 100],
            "LineTotal": [10.0, 30.0, 20.0, 50.0, 100.0],
        }
    )
    return keys, orders, lines


def test_shares_exclude_future_and_include_history_start():
    profiles = historical_product_shares(*source())
    annual = profiles[profiles.scope.eq("annual")].set_index("ProductID")
    latest = profiles[profiles.scope.eq("latest_order")].set_index("ProductID")
    assert set(profiles.ProductID) == {7, 8}
    np.testing.assert_allclose(annual.loc[[7, 8], "share"], [0.5, 0.5])
    np.testing.assert_allclose(latest.loc[[7, 8], "share"], [2 / 7, 5 / 7])


def test_source_reconciliation_and_missing_cases_fail():
    keys, orders, lines = source()
    lines.loc[2, "LineTotal"] = 21
    with pytest.raises(AssertionError):
        historical_product_shares(keys, orders, lines)
    keys.loc[1] = [2, pd.Timestamp("2013-04-01")]
    with pytest.raises(ValueError, match="Missing eligible"):
        historical_product_shares(keys, orders, source()[2])


def test_matched_distances_and_neighbor_presence():
    profiles = pd.DataFrame(
        {
            "StoreID": [1, 2, 3],
            "snapshot": pd.to_datetime(["2013-04-01"] * 3),
            "ProductID": [7, 7, 8],
            "share": [1.0, 1.0, 1.0],
            "scope": "annual",
        }
    )
    matches = pd.DataFrame(
        {
            "StoreID": [1, 1],
            "snapshot": pd.to_datetime(["2013-04-01"] * 2),
            "control_StoreID": [2, 3],
            "comparison": "example",
            "k": 3,
            "distance": [0.1, 0.2],
        }
    )
    summary, distances = matched_sku_tables(profiles, matches)
    np.testing.assert_allclose(distances.product_total_variation, [0.0, 1.0])
    overall = summary[summary.aggregation.eq("overall")].set_index("ProductID")
    assert overall.loc[7, "mean_difference"] == 0.5
    assert overall.loc[7, "control_mean_positive_fraction"] == 0.5


def test_vocabulary_counts_training_resellers_and_maps_unseen_to_other():
    from comparisons.product_mix_comparison import (
        add_product_shares,
        product_vocabulary,
    )

    cases = pd.DataFrame(
        {"StoreID": range(1, 22), "snapshot": pd.Timestamp("2013-04-01")}
    )
    rows = []
    for scope in ("annual", "latest_order"):
        for store in range(1, 22):
            rows.append(
                dict(
                    StoreID=store,
                    snapshot=pd.Timestamp("2013-04-01"),
                    scope=scope,
                    ProductID=7 if store <= 20 else 99,
                    share=1.0,
                )
            )
        for date in ("2012-07-01", "2012-10-01"):
            rows.append(
                dict(
                    StoreID=1,
                    snapshot=pd.Timestamp(date),
                    scope=scope,
                    ProductID=8,
                    share=1.0,
                )
            )
    profiles = pd.DataFrame(rows)
    training_keys = pd.concat(
        [
            cases.iloc[:20],
            pd.DataFrame(
                {
                    "StoreID": [1, 1],
                    "snapshot": pd.to_datetime(["2012-07-01", "2012-10-01"]),
                }
            ),
        ]
    )
    vocabulary, support = product_vocabulary(training_keys, profiles)
    assert vocabulary == [7]
    repeated, repeated_support = product_vocabulary(
        pd.concat([training_keys, training_keys]), profiles
    )
    assert repeated == vocabulary
    pd.testing.assert_frame_equal(support, repeated_support)
    assert support.set_index("ProductID").loc[8, "resellers"] == 1
    assert 99 not in set(support.ProductID)
    expanded, names = add_product_shares(cases.iloc[[20]], profiles, vocabulary)
    assert names == [
        "annual_sku_7",
        "annual_sku_other",
        "latest_order_sku_7",
        "latest_order_sku_other",
    ]
    np.testing.assert_allclose(expanded[names], [[0, 1, 0, 1]])


def test_cluster_resampling_carries_snapshots_and_multiplicity():
    from comparisons.product_mix_comparison import bootstrap_training_rows

    train = pd.DataFrame(
        {
            "StoreID": np.repeat(range(1, 21), 2),
            "snapshot": pd.to_datetime(["2013-01-01", "2013-04-01"] * 20),
            "churn": [0, 1] * 20,
        }
    )
    sampled, draws = bootstrap_training_rows(train, 42)
    assert len(draws) == 20
    assert set(sampled.StoreID).issubset(train.StoreID)
    for store, count in pd.Series(draws).value_counts().items():
        assert (
            sampled[sampled.StoreID.eq(store)]
            .groupby("snapshot")
            .size()
            .eq(count)
            .all()
        )
    assert sampled.churn.value_counts().to_dict() == {0: 20, 1: 20}


def test_bagging_averages_all_fixed_replicas_and_keeps_identities(monkeypatch):
    from comparisons import product_mix_comparison as module

    train = pd.DataFrame({"StoreID": np.repeat(range(1, 21), 2), "churn": [0, 1] * 20})
    valid = pd.DataFrame({"StoreID": [100]})
    seen = []

    def fake_fit(sampled, validation, profiles, seeds):
        assert set(sampled.StoreID).isdisjoint(validation.StoreID)
        seen.append(seeds[0])
        score = len(seen) / 10
        return np.array([score]), [{"seed": seeds[0]}]

    monkeypatch.setattr(module, "fit_product_mix", fake_fit)
    scores, identities = module.fit_bagged_product_mix(train, valid, pd.DataFrame())
    np.testing.assert_allclose(scores, [0.3])
    assert seen == list(module.study.MODEL_SEEDS)
    assert len(identities) == 5
    assert all(len(row["bootstrap_store_ids"]) == 20 for row in identities)
    assert all(row["candidate"] == "cadence_product_mix_bagged" for row in identities)

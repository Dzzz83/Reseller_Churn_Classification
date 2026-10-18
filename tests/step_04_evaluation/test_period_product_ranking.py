"""Product-engagement provenance, tied ranks and nested period-weighted execution."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.period_product_ranking import (
    build_features,
    fit_arm,
    weighted_rank_metrics,
)
from comparisons.period_weighting import period_weights
from comparisons.reseller_weighting import fit_weighted
from reseller_churn.artifacts import write_json, write_manifest


def test_count_and_density_features_preserve_cases_and_reject_future_or_missing_orders():
    data = pd.DataFrame({name: [1.0, 1.0] for name in study.FEATURES})
    data["StoreID"] = [1, 2]
    data["snapshot"] = pd.Timestamp("2013-01-01")
    data["churn"] = [0, 1]
    history = data[study.KEYS].assign(unique_products_12m=2, n_orders_12m=2)
    profiles = pd.DataFrame(
        [
            dict(
                StoreID=store,
                snapshot=pd.Timestamp("2013-01-01"),
                scope=scope,
                ProductID=product,
                share=0.5,
            )
            for store in [1, 2]
            for scope in ["annual", "latest_order"]
            for product in [10, 20]
        ]
    )
    orders = pd.DataFrame(
        [
            dict(
                StoreID=store,
                snapshot=pd.Timestamp("2013-01-01"),
                SalesOrderID=store * 10 + i,
                OrderDate=pd.Timestamp(date),
            )
            for store in [1, 2]
            for i, date in enumerate(["2012-05-01", "2012-12-01"])
        ]
    )
    result = build_features(data, history, profiles, orders)
    assert (
        len(result) == 2
        and result.unique_products_12m.eq(2).all()
        and result.unique_products_per_order_12m.eq(1).all()
    )
    with pytest.raises(ValueError, match="Future"):
        build_features(
            data, history, profiles, orders.assign(OrderDate=pd.Timestamp("2013-01-01"))
        )
    with pytest.raises(ValueError, match="Missing"):
        build_features(data, history, profiles, orders[orders.StoreID.eq(1)])


def test_weighted_rank_bootstrap_matches_sklearn_with_ties_and_zero_weights():
    y = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.8, 0.5, 0.5, 0.1, 0.1])
    weights = np.array(
        [[1, 1, 1, 1, 1, 1], [2, 0, 1, 3, 0, 1], [0, 1, 0, 2, 0, 1], [0, 0, 0, 0, 0, 0]]
    )
    ap, auc = weighted_rank_metrics(y, scores, weights)
    for i in [0, 1]:
        assert ap[i] == pytest.approx(
            average_precision_score(y, scores, sample_weight=weights[i])
        )
        assert auc[i] == pytest.approx(
            roc_auc_score(y, scores, sample_weight=weights[i])
        )
    assert np.isnan(ap[2:]).all() and np.isnan(auc[2:]).all()


def test_synthetic_nested_fold_persists_models_and_uses_training_only_cutoffs(tmp_path):
    data = study.synthetic_development()
    data["unique_products_12m"] = np.arange(len(data)) % 7 + 1
    data["n_orders_12m"] = 4
    data["unique_products_per_order_12m"] = data.unique_products_12m / 4
    ti, vi = study.grouped_splits(data, 5, 301)[0]
    split = dict(
        repeat=301,
        outer_fold=0,
        train_rows=ti.tolist(),
        validation_rows=vi.tolist(),
        train_store_ids=sorted(data.iloc[ti].StoreID.unique().tolist()),
        validation_store_ids=sorted(data.iloc[vi].StoreID.unique().tolist()),
    )
    train, held = data.iloc[ti], data.iloc[vi]
    spec = dict(
        id="period_class_weighted",
        features=study.FEATURES,
        parameters={**study.BASE_PARAMETERS, "max_iter": 5},
    )
    source = tmp_path / "cache"
    folder = source / "repeat_301/fold_0"
    folder.mkdir(parents=True)
    parts = []
    records = []
    for fold, (a, b) in enumerate(study.grouped_splits(train, 3, 78)):
        scores, identity, _, _ = fit_weighted(
            train.iloc[a],
            train.iloc[b][study.FEATURES],
            spec,
            weight_function=period_weights,
        )
        parts.append(
            train.iloc[b][study.KEYS + ["churn"]].assign(
                candidate=spec["id"], score=scores, inner_fold=fold
            )
        )
        records.extend([dict(**r, stage="inner", inner_fold=fold) for r in identity])
    pd.concat(parts, ignore_index=True).to_csv(folder / "inner_scores.csv", index=False)
    score, identity, _, _ = fit_weighted(
        train, held[study.FEATURES], spec, weight_function=period_weights
    )
    held[study.KEYS + ["churn", "n_orders_3m"]].assign(
        strategy=spec["id"], score=score, cutoff=0.1, repeat=301, outer_fold=0
    ).to_csv(folder / "predictions.csv", index=False)
    write_json(
        folder / "model_identities.json",
        records + [dict(**r, stage="outer") for r in identity],
    )
    write_manifest(folder, dict(synthetic=True))
    candidate = dict(
        id="synthetic_breadth",
        features=study.FEATURES + ["unique_products_12m"],
        parameters=spec["parameters"],
    )
    output = tmp_path / "new"
    output.mkdir()
    predicted, feasible = fit_arm(
        data, pd.DataFrame(), source, split, candidate, output
    )
    assert len(predicted) == 2 * len(held)
    models = joblib.load(output / "outer_models.joblib")
    expected = np.mean(
        [m.predict_proba(held[candidate["features"]])[:, 1] for m in models], axis=0
    )
    np.testing.assert_array_equal(
        predicted[predicted.strategy.eq(candidate["id"])].score, expected
    )
    records = json.loads((output / "model_identities.json").read_text())["fits"]
    assert len(records) == 20 and len(models) == 5
    inner = pd.read_csv(output / "inner_scores.csv")
    assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()
    assert set(inner.StoreID).isdisjoint(held.StoreID)
    assert isinstance(feasible, bool)

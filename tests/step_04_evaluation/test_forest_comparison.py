"""The new forest must learn imputation from training data alone."""

import numpy as np

from comparisons.forest_comparison import fit_forest
from comparisons.grouped_precision_recall import FEATURES, synthetic_development
from reseller_churn.config.model_settings import RandomForestSettings


def test_forest_imputer_excludes_validation_and_keeps_seed_averaging():
    data = synthetic_development()
    train = data.iloc[:180].copy()
    train.loc[train.index[:30], "discount_change_3m"] = np.nan
    validation = data.iloc[180:185][FEATURES].copy()
    validation["discount_change_3m"] = 1000.0
    scores, identities = fit_forest(
        train, validation, seeds=(42, 78), settings=RandomForestSettings(n_estimators=8)
    )
    assert len(scores) == len(validation)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    expected = train[FEATURES].median().to_numpy()
    for identity in identities:
        np.testing.assert_allclose(
            identity["imputer_statistics"], expected, rtol=0, atol=0
        )
        assert (
            identity["missing_indicator_columns"]
            == np.flatnonzero(train[FEATURES].isna().any()).tolist()
        )
        assert identity["training_rows"] == len(train)
        assert identity["training_positives"] == train.churn.sum()
        assert identity["forest_parameters"]["class_weight"] is None
        assert identity["forest_parameters"]["n_jobs"] == 1
    assert {record["seed"] for record in identities} == {42, 78}


def test_forest_fold_reuses_baseline_and_preserves_inner_groups(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from comparisons import forest_comparison as forest
    from comparisons.basket_breadth import evaluate_fold as make_source
    from comparisons.grouped_precision_recall import grouped_splits

    data = synthetic_development()
    data["last_order_lines"] = 1.0
    ti, vi = grouped_splits(data, 5, 42)[0]
    split = dict(
        repeat=42,
        outer_fold=0,
        train_rows=ti.tolist(),
        validation_rows=vi.tolist(),
        train_store_ids=sorted(data.iloc[ti].StoreID.unique().tolist()),
        validation_store_ids=sorted(data.iloc[vi].StoreID.unique().tolist()),
    )
    source = tmp_path / "source"
    original = make_source(data, split, source)
    actual_fit = forest.fit_forest
    monkeypatch.setattr(
        forest,
        "fit_forest",
        lambda train, valid: actual_fit(
            train, valid, seeds=(42, 78), settings=RandomForestSettings(n_estimators=8)
        ),
    )
    out = tmp_path / "comparison"
    predicted = forest.evaluate_fold(data, split, source / "repeat_42/fold_0", out)
    assert set(predicted.strategy) == {"baseline", forest.FOREST, forest.BLEND}
    baseline = predicted[predicted.strategy.eq("baseline")].reset_index(drop=True)
    expected = original[original.strategy.eq("baseline")].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, expected)
    inner = pd.read_csv(out / "inner_scores.csv")
    assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()
    assert set(inner.StoreID).isdisjoint(data.iloc[vi].StoreID)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["new_forest_fits"] == 8
    assert manifest["reused_hgb_fits"] == 20

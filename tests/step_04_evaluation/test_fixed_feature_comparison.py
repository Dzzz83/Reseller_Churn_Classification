"""A new numeric feature keeps the original cached baseline and nested groups."""

import json

import numpy as np
import pandas as pd

from comparisons.basket_breadth import evaluate_fold as make_source
from comparisons.fixed_feature_comparison import evaluate_fixed_fold
from comparisons.grouped_precision_recall import (
    BASE_PARAMETERS,
    FEATURES,
    grouped_splits,
    synthetic_development,
)


def test_fixed_feature_fold_preserves_baseline_and_inner_groups(tmp_path):
    data = synthetic_development()
    data["last_order_lines"] = 1.0
    data["last_order_unit_revenue"] = np.linspace(1, 100, len(data))
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
    expected = make_source(data, split, source)
    spec = dict(
        id="unit_revenue",
        features=FEATURES + ["overdue_ratio", "gap_cv", "last_order_unit_revenue"],
        parameters={
            **BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
            "max_iter": 10,
        },
    )
    out = tmp_path / "new"
    predicted = evaluate_fixed_fold(data, split, source / "repeat_42/fold_0", out, spec)
    pd.testing.assert_frame_equal(
        predicted[predicted.strategy.eq("baseline")].reset_index(drop=True),
        expected[expected.strategy.eq("baseline")].reset_index(drop=True),
    )
    inner = pd.read_csv(out / "inner_scores.csv")
    assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()
    assert set(inner.StoreID).isdisjoint(data.iloc[vi].StoreID)
    records = json.loads((out / "model_identities.json").read_text())
    assert len(records) == 40
    assert sum(row["candidate"] == "unit_revenue" for row in records) == 20
    assert np.isfinite(predicted.score).all()

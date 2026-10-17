"""Operational acceptance and reseller-cluster inference safeguards."""

import numpy as np
import pandas as pd
import pytest

from comparisons.basket_breadth import ARMS, evaluate_fold
from comparisons.grouped_precision_recall import (
    REPEAT_SEEDS,
    grouped_splits,
    synthetic_development,
)
from comparisons.matched_evaluation import acceptance, paired_bootstrap


def decision_fixture(candidate_tp=(10, 10, 10), candidate_fp=(80, 80, 80)):
    rows = []
    for arm in ARMS:
        for i, repeat in enumerate(REPEAT_SEEDS):
            row = {
                "policy": arm,
                "repeat": repeat,
                "scope": "repeat_pooled",
                "slice": "overall",
                "group": "all",
                "tp": 10 if arm == "baseline" else candidate_tp[i],
                "fp": 100 if arm == "baseline" else candidate_fp[i],
            }
            row["recall"] = row["tp"] / 12
            rows.extend([row, {**row, "slice": "activity", "group": "recent_purchase"}])
    return pd.DataFrame(rows)


def test_acceptance_rejects_recall_loss_hidden_by_mean():
    bootstrap = pd.DataFrame(
        {"recall_difference": [0.01] * 10, "fp_reduction": [20.0] * 10}
    )
    decision, _, _ = acceptance(decision_fixture((11, 10, 9)), bootstrap)
    assert not decision["accepted"]
    assert not decision["checks"]["recall_preserved_every_repeat"]
    decision, _, _ = acceptance(decision_fixture(), bootstrap)
    assert decision["accepted"]
    bootstrap["recall_difference"] = -0.001
    assert not acceptance(decision_fixture(), bootstrap)[0]["accepted"]


def test_bootstrap_preserves_identical_arms_and_repeat_dependence():
    rows = []
    for arm in ARMS:
        for repeat in REPEAT_SEEDS:
            for store, churn in [(1, 1), (2, 0), (3, 1)]:
                for snapshot in ["2012-07-01", "2013-04-01"]:
                    rows.append(
                        {
                            "StoreID": store,
                            "snapshot": snapshot,
                            "repeat": repeat,
                            "churn": churn,
                            "score": 0.8,
                            "cutoff": 0.5,
                            "strategy": arm,
                        }
                    )
    predictions = pd.DataFrame(rows)
    null = paired_bootstrap(predictions, draws=100)
    assert null.recall_difference.dropna().eq(0).all()
    assert null.fp_reduction.eq(0).all()
    # Removing one reseller's two false alerts changes each repeat identically;
    # repeated validation must not multiply the reduction by three.
    predictions.loc[
        (predictions.strategy == "basket_breadth") & (predictions.StoreID == 2), "score"
    ] = 0.1
    draws = paired_bootstrap(predictions, draws=100)
    assert set(draws.fp_reduction.unique()).issubset({0, 2, 4, 6})
    assert draws.recall_difference.dropna().eq(0).all()


@pytest.mark.parametrize("feature", ["last_order_lines", "unique_products_12m"])
def test_nested_synthetic_fold_saves_two_matched_arms(tmp_path, feature):
    data = synthetic_development()
    data[feature] = np.arange(len(data)) % 4 + 1
    ti, vi = grouped_splits(data, 5, 42)[0]
    split = {
        "repeat": 42,
        "outer_fold": 0,
        "train_rows": ti.tolist(),
        "validation_rows": vi.tolist(),
        "train_store_ids": sorted(data.iloc[ti].StoreID.unique().tolist()),
        "validation_store_ids": sorted(data.iloc[vi].StoreID.unique().tolist()),
    }
    predictions = evaluate_fold(data, split, tmp_path, feature)
    assert set(predictions.strategy) == set(ARMS)
    for _, group in predictions.groupby("strategy"):
        assert len(group) == len(vi)
        assert group.score.between(0, 1).all()
    inner = pd.read_csv(tmp_path / "repeat_42/fold_0/inner_scores.csv")
    assert set(inner.StoreID).isdisjoint(data.iloc[vi].StoreID)
    assert inner.groupby("StoreID").inner_fold.nunique().eq(1).all()


def test_independent_cutoff_preserves_tied_scores():
    from comparisons.verify_basket_breadth import independent_cutoff

    assert independent_cutoff(np.array([1, 1, 0]), np.array([0.9, 0.2, 0.2])) == 0.2
    assert independent_cutoff(np.array([1, 0]), np.array([0.0, 0.5])) == 0.0


def test_transfer_variation_reports_nonadditive_cancellation():
    from comparisons.cutoff_transfer_diagnosis import decision_variation

    data = pd.DataFrame(
        {"score_a": [0.5], "cutoff_a": [0.5], "score_b": [0.3], "cutoff_b": [0.3]}
    )
    result = decision_variation(data)
    assert result["actual_disagreement"] == 0
    assert result["score_only_disagreement"] == 0.5
    assert result["cutoff_only_disagreement"] == 0.5


def test_paired_analysis_supports_new_partitions_and_named_candidate():
    metrics = decision_fixture().replace({"basket_breadth": "cadence_no_count"})
    bootstrap = pd.DataFrame(
        {"recall_difference": [0.01] * 20, "fp_reduction": [10.0] * 20}
    )
    assert acceptance(metrics, bootstrap, candidate_id="cadence_no_count")[0][
        "accepted"
    ]
    rows = []
    for arm in ["baseline", "cadence_no_count"]:
        for repeat in [301, 919, 1201]:
            for store, y in [(1, 1), (2, 0)]:
                rows.append(
                    dict(
                        strategy=arm,
                        repeat=repeat,
                        StoreID=store,
                        churn=y,
                        score=0.8,
                        cutoff=0.5,
                    )
                )
    samples = paired_bootstrap(
        pd.DataFrame(rows), draws=50, candidate_id="cadence_no_count"
    )
    assert samples.recall_difference.dropna().eq(0).all()
    assert samples.fp_reduction.eq(0).all()

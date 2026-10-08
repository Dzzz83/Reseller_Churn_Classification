"""Regression checks for the notebook-faithful corrected reproduction."""
import numpy as np
import pandas as pd
import pytest

from comparisons.correlation_pruned_churn.corrected_model import _is_log_feature
from comparisons.correlation_pruned_churn import __path__
from importlib import util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "comparisons" / "correlation_pruned_churn" / "05_reproduce_corrected_original.py"
spec = util.spec_from_file_location("friend_original_replication", SCRIPT)
module = util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_f1_threshold_matches_original_notebook_rule():
    labels = np.array([0, 1, 1, 0, 1, 0])
    probabilities = np.array([0.02, 0.61, 0.31, 0.12, 0.47, 0.28])
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(labels, probabilities)
    scores = 2 * p * r / np.maximum(p + r, 1e-12)
    expected = t[np.argmax(scores[:-1])]
    assert module.threshold_maximizing_oof_f1(labels, probabilities) == expected


def test_recall_changes_with_threshold_but_pr_auc_does_not():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.3, 0.2, 0.6])
    low = module.classification_metrics(y, probabilities, 0.25)
    high = module.classification_metrics(y, probabilities, 0.5)
    assert low["recall"] > high["recall"]
    assert low["pr_auc"] == high["pr_auc"]


def test_original_log_columns_also_apply_to_unverified_profile_diagnostic():
    for column in ("AnnualSales", "AnnualRevenue", "NumberEmployees", "avg_qty", "rev_trend"):
        assert _is_log_feature(column)


def test_published_comparison_contains_only_relevant_profile_scope():
    frame = pd.DataFrame([
        {"variant": "full", "feature_scope": "historical_only", "model": "RandomForest",
         "cv_pr_auc": 0.7, "cv_recall": 0.7, "test_pr_auc": 0.8, "test_recall": 0.8},
        {"variant": "full", "feature_scope": "unverified_store_profile_diagnostic",
         "model": "RandomForest", "cv_pr_auc": 0.55, "cv_recall": 0.68,
         "test_pr_auc": 0.40, "test_recall": 0.80},
    ])
    result = module.make_published_comparison(frame)
    assert len(result) == 1
    assert result.iloc[0]["original_cv_pr_auc"] == 0.500
    assert result.iloc[0]["cv_recall_delta"] == pytest.approx(-0.025)


def test_training_snapshots_end_before_retrospective_test():
    assert len(module.TRAIN_SNAPSHOTS) == 5
    assert module.RETROSPECTIVE_TEST == "2013-11-01"
    assert "2013-10-01" not in module.TRAIN_SNAPSHOTS
    for snapshot in module.TRAIN_SNAPSHOTS:
        assert pd.Timestamp(snapshot) + pd.DateOffset(months=6) <= pd.Timestamp(
            module.RETROSPECTIVE_TEST
        )

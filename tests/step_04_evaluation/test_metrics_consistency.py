"""Shared metric calculations must match their established definitions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from reseller_churn.evaluation.metrics import (
    calculate_ranking_metrics,
    calculate_threshold_metrics,
)


def test_shared_ranking_metrics_match_sklearn():
    targets = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.5, 0.4, 0.6, 0.8]

    metrics = calculate_ranking_metrics(targets, scores)
    assert metrics["pr_auc"] == average_precision_score(targets, scores)
    assert metrics["roc_auc"] == roc_auc_score(targets, scores)


def test_threshold_metrics_count_exact_boundary_as_positive():
    targets = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.49, 0.8, 0.9])
    predicted = (scores >= 0.5).astype(int)

    metrics = calculate_threshold_metrics(targets, scores, threshold=0.5)
    assert metrics == {
        "tp": 2,
        "fp": 2,
        "fn": 1,
        "tn": 1,
        "precision": precision_score(targets, predicted, zero_division=0),
        "recall": recall_score(targets, predicted, zero_division=0),
        "f1": f1_score(targets, predicted, zero_division=0),
        "accuracy": accuracy_score(targets, predicted),
        "alert_count": 4,
        "alert_rate": 4 / 6,
    }
    assert predicted[1] == predicted[2] == 1


def test_average_precision_has_accurate_name_and_legacy_alias():
    metrics = calculate_ranking_metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.6])
    assert metrics["average_precision"] == metrics["pr_auc"]


def test_undefined_ranking_metrics_are_explicit_for_one_class_targets():
    metrics = calculate_ranking_metrics([0, 0, 0], [0.1, 0.4, 0.8])
    assert metrics == {
        "average_precision": None,
        "pr_auc": None,
        "roc_auc": None,
    }


def test_zero_predictions_keep_confusion_counts_and_zero_division_metrics():
    metrics = calculate_threshold_metrics([1, 0, 0], [0.2, 0.1, 0.3], 0.5)
    assert metrics == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 2,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 2 / 3,
        "alert_count": 0,
        "alert_rate": 0.0,
    }

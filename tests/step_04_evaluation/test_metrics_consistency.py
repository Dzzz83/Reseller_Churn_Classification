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
        "precision": precision_score(targets, predicted, zero_division=0),
        "recall": recall_score(targets, predicted, zero_division=0),
        "f1": f1_score(targets, predicted, zero_division=0),
        "accuracy": accuracy_score(targets, predicted),
    }
    assert predicted[1] == predicted[2] == 1

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_ranking_metrics(
    target,
    probabilities,
) -> dict[str, float]:
    return {
        "pr_auc": average_precision_score(
            target,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            target,
            probabilities,
        ),
    }


def calculate_threshold_metrics(
    target,
    probabilities,
    threshold: float,
) -> dict[str, float]:
    predictions = (
        np.asarray(probabilities)
        >= threshold
    ).astype(int)

    return {
        "precision": precision_score(
            target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            target,
            predictions,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            target,
            predictions,
        ),
    }

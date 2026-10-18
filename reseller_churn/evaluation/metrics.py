import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_ranking_metrics(
    target,
    probabilities,
) -> dict[str, float | None]:
    target_values = np.asarray(target)
    probability_values = np.asarray(probabilities, dtype=float)
    if (
        target_values.ndim != 1
        or probability_values.ndim != 1
        or len(target_values) != len(probability_values)
        or len(target_values) == 0
    ):
        raise ValueError("Target and probabilities must have equal, non-zero length.")
    if not np.isin(target_values, [0, 1]).all():
        raise ValueError("Target must contain only binary 0/1 labels.")
    if (
        not np.isfinite(probability_values).all()
        or ((probability_values < 0) | (probability_values > 1)).any()
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    positive_count = int(np.count_nonzero(target_values == 1))
    average_precision = (
        float(average_precision_score(target_values, probability_values))
        if positive_count
        else None
    )
    roc_auc = (
        float(roc_auc_score(target_values, probability_values))
        if len(np.unique(target_values)) == 2
        else None
    )
    return {
        "average_precision": average_precision,
        # Compatibility alias for existing reports and regression checks.
        "pr_auc": average_precision,
        "roc_auc": roc_auc,
    }


def calculate_threshold_metrics(
    target,
    probabilities,
    threshold: float,
) -> dict[str, float | int]:
    raw_target = np.asarray(target)
    target_values = raw_target.astype(int)
    probability_values = np.asarray(probabilities, dtype=float)
    if (
        target_values.ndim != 1
        or probability_values.ndim != 1
        or len(target_values) != len(probability_values)
        or len(target_values) == 0
    ):
        raise ValueError("Target and probabilities must have equal, non-zero length.")
    if not np.isin(raw_target, [0, 1]).all():
        raise ValueError("Target must contain only binary 0/1 labels.")
    if (
        not np.isfinite(probability_values).all()
        or ((probability_values < 0) | (probability_values > 1)).any()
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1.")

    predictions = (probability_values >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target_values, predictions, labels=[0, 1]
    ).ravel()
    true_negative = int(true_negative)
    false_positive = int(false_positive)
    false_negative = int(false_negative)
    true_positive = int(true_positive)
    alert_count = true_positive + false_positive

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": float(
            precision_score(target_values, predictions, zero_division=0)
        ),
        "recall": float(recall_score(target_values, predictions, zero_division=0)),
        "f1": float(f1_score(target_values, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target_values, predictions)),
        "alert_count": alert_count,
        "alert_rate": float(alert_count / len(target_values)),
    }

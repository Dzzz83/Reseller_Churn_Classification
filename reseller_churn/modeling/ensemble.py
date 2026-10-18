"""Probability aggregation shared by saved ensembles and research fits."""

import numpy as np


def mean_probabilities(models, features):
    if not models:
        raise ValueError("An ensemble must contain at least one estimator.")
    scores = []
    for model in models:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.shape != (len(features), 2):
            raise ValueError("Expected two probability columns for every feature row.")
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Estimator probabilities must be finite values in [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("Estimator class probabilities must sum to one.")
        scores.append(probabilities[:, 1])
    return np.mean(scores, axis=0)


def apply_threshold(scores, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one.")
    values = np.asarray(scores, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Scores must be a finite probability vector.")
    return (values >= threshold).astype(int)

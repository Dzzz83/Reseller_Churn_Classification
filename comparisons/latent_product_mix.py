"""Training-only low-dimensional representations of accepted SKU revenue shares."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import require

KEYS = study.KEYS
BASE_FEATURES = study.FEATURES + ["overdue_ratio", "gap_cv"]
CANDIDATE = "cadence_latent_mix"


def product_matrix(keys, profiles, scope, vocabulary):
    source = (
        profiles[profiles.scope.eq(scope)]
        .pivot(index=KEYS, columns="ProductID", values="share")
        .fillna(0)
    )
    source = source.reindex(pd.MultiIndex.from_frame(keys[KEYS]))
    require(source.notna().all().all(), "Missing purchase-mix profile")
    matrix = source.reindex(columns=vocabulary, fill_value=0).to_numpy()
    unseen = np.clip(1 - matrix.sum(axis=1), 0, 1)
    return matrix, unseen


def learn_projection(keys, profiles):
    training = profiles[profiles.scope.eq("annual") & profiles.share.gt(0)].merge(
        keys[KEYS], on=KEYS, validate="many_to_one"
    )
    vocabulary = sorted(training.ProductID.astype(int).unique().tolist())
    matrix, _ = product_matrix(keys, profiles, "annual", vocabulary)
    require(
        min(matrix.shape) >= 5, "Insufficient training rows/SKUs for frozen projection"
    )
    pca = PCA(n_components=5, svd_solver="full").fit(matrix)
    return dict(
        vocabulary=vocabulary,
        mean=pca.mean_.tolist(),
        components=pca.components_.tolist(),
        singular_values=pca.singular_values_.tolist(),
        explained_variance=pca.explained_variance_.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )


def project_cases(cases, profiles, projection):
    result = cases.copy()
    columns = []
    components, center = (
        np.array(projection["components"]),
        np.array(projection["mean"]),
    )
    for scope in ("annual", "latest_order"):
        matrix, unseen = product_matrix(
            cases[KEYS], profiles, scope, projection["vocabulary"]
        )
        names = [f"{scope}_mix_pc{index + 1}" for index in range(5)]
        result[names] = (matrix - center) @ components.T
        name = f"{scope}_unseen_share"
        result[name] = unseen
        columns.extend([*names, name])
    return result, columns


def fit_latent_mix(train, validation_features, profiles):
    projection = learn_projection(train[KEYS], profiles)
    expanded, extra = project_cases(train, profiles, projection)
    validation, _ = project_cases(validation_features, profiles, projection)
    spec = dict(
        id=CANDIDATE,
        features=BASE_FEATURES + extra,
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    scores, identities, spread, _ = study.fit_scores(
        expanded, validation[spec["features"]], spec
    )
    for identity in identities:
        identity.update(
            projection=projection, features=spec["features"], seed_score_range=spread
        )
    return scores, identities

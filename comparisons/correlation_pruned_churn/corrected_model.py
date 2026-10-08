"""Leakage-safe preprocessing and fold-local feature selection.

Faithful to the friend's modeling choices except historically unverifiable
profile variables are excluded and supervised feature selection is fitted
separately on each training fold, never on its validation labels.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from comparisons.correlation_pruned_churn.corrected_features import (
    CATEGORICAL_FEATURES,
)


def select_numeric_features(
    training: pd.DataFrame,
    numeric_candidates: list[str],
) -> list[str]:
    """Replicate correlation pruning using TRAINING rows and labels only."""
    nonconstant = [
        column
        for column in numeric_candidates
        if training[column].nunique(dropna=True) > 1
    ]

    correlations = training[nonconstant].corr(
        method="spearman"
    )
    correlations_with_churn = (
        training[nonconstant]
        .corrwith(training["churn"], method="spearman")
        .abs()
        .fillna(0)
    )

    pairs = []
    for index, left in enumerate(nonconstant):
        for right in nonconstant[index + 1:]:
            rho = correlations.loc[left, right]
            if pd.notna(rho) and abs(rho) > 0.8:
                pairs.append((abs(rho), left, right))

    pairs.sort(reverse=True)
    dropped: set[str] = set()

    for _, left, right in pairs:
        if left in dropped or right in dropped:
            continue

        if (
            correlations_with_churn[left]
            < correlations_with_churn[right]
        ):
            dropped.add(left)
        else:
            dropped.add(right)

    selected = [
        column for column in nonconstant
        if column not in dropped
    ]
    if not selected:
        raise AssertionError("Feature pruning removed every numeric feature")
    return selected


def _is_log_feature(name: str) -> bool:
    return (
        name.startswith(("revenue_", "n_orders_"))
        or name in (
            "aov",
            "avg_qty",
            "avg_lines",
            "rev_trend",
            "AnnualSales",
            "AnnualRevenue",
            "NumberEmployees",
        )
    )


def create_correlation_pruned_pipeline(
    numeric_features: list[str],
    model_name: str,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> Pipeline:
    log_features = [
        feature for feature in numeric_features
        if _is_log_feature(feature)
    ]
    other_features = [
        feature for feature in numeric_features
        if feature not in log_features
    ]

    transforms = []

    if log_features:
        transforms.append(
            (
                "log_numeric",
                make_pipeline(
                    FunctionTransformer(
                        np.log1p,
                        feature_names_out="one-to-one",
                    ),
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                    StandardScaler(),
                ),
                log_features,
            )
        )

    if other_features:
        transforms.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                    StandardScaler(),
                ),
                other_features,
            )
        )

    transforms.append(
        (
            "categorical",
            make_pipeline(
                SimpleImputer(
                    strategy="most_frequent",
                ),
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
            list(categorical_features),
        )
    )

    if model_name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "LogReg":
        estimator = LogisticRegression(
            max_iter=5000,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transforms),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def fit_correlation_pruned_model(
    train: pd.DataFrame,
    numeric_candidates: list[str],
    model_name: str,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[Pipeline, list[str]]:
    selected_numeric = select_numeric_features(
        train, numeric_candidates
    )
    selected_features = (
        selected_numeric + list(categorical_features)
    )

    for col in selected_numeric:
        if _is_log_feature(col) and (
            train[col].dropna() < -1
        ).any():
            raise ValueError(f"Negative log-input feature: {col}")

    model = create_correlation_pruned_pipeline(
        numeric_features=selected_numeric,
        model_name=model_name,
        categorical_features=categorical_features,
    )
    model.fit(
        train[selected_features],
        train["churn"].astype(int),
    )
    return model, selected_features

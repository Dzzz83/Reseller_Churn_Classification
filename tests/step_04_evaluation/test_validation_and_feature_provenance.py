import pandas as pd
import pytest

from comparisons.correlation_pruned_churn.feature_provenance import (
    attach_feature_scope,
)
from comparisons.correlation_pruned_churn.validation import (
    DEVELOPMENT_SNAPSHOTS,
    ValidationPlans,
)


def sample_development_frame() -> pd.DataFrame:
    rows = []
    for snapshot in DEVELOPMENT_SNAPSHOTS:
        for store_id in range(1, 11):
            rows.append(
                {
                    "StoreID": store_id,
                    "snapshot": snapshot,
                    "churn": int(store_id % 2 == 0),
                    "recency_days": store_id,
                }
            )
    return pd.DataFrame(rows)


def test_group_kfold_never_reuses_store_across_fold_sides() -> None:
    data = sample_development_frame()
    folds = ValidationPlans.group_kfold(data)

    assert len(folds) == 5
    all_validation = []

    for fold in folds:
        train_ids = set(
            data.iloc[fold.training_indices]["StoreID"]
        )
        valid_ids = set(
            data.iloc[fold.validation_indices]["StoreID"]
        )
        assert train_ids.isdisjoint(valid_ids)
        all_validation.extend(fold.validation_indices.tolist())

    assert sorted(all_validation) == list(range(len(data)))


def test_temporal_folds_only_use_known_training_labels() -> None:
    data = sample_development_frame()
    folds = ValidationPlans.temporal(data)

    assert len(folds) == 2
    for fold in folds:
        training = data.iloc[fold.training_indices]
        validation = data.iloc[fold.validation_indices]

        assert training["snapshot"].max() < validation["snapshot"].min()

        last_known = (
            pd.to_datetime(training["snapshot"])
            + pd.DateOffset(months=6)
        )
        assert (
            last_known <= pd.Timestamp(validation["snapshot"].min())
        ).all()


def test_protected_test_snapshot_is_rejected() -> None:
    data = sample_development_frame()
    data.loc[0, "snapshot"] = "2013-10-01"

    with pytest.raises(AssertionError):
        ValidationPlans.verify_development_period(data)


def test_profile_features_are_not_silently_declared_safe() -> None:
    data = sample_development_frame()
    stores = pd.DataFrame(
        {
            "StoreID": range(1, 11),
            "SalesPersonID": [275] * 10,
            "AnnualSales": [800000] * 10,
            "AnnualRevenue": [80000] * 10,
            "SquareFeet": [10000] * 10,
            "NumberEmployees": [12] * 10,
            "BusinessType": ["BM"] * 10,
            "Specialty": ["Road"] * 10,
            "Brands": ["2"] * 10,
            "Internet": ["DSL"] * 10,
        }
    )

    safe_frame, safe_scope = attach_feature_scope(
        data, stores, include_unverified_profile=False
    )
    assert safe_scope.name == "historical_only"
    assert "AnnualSales" not in safe_frame
    assert "AnnualSales" not in safe_scope.numeric_candidates

    diagnostic_frame, diagnostic_scope = attach_feature_scope(
        data, stores, include_unverified_profile=True
    )
    assert diagnostic_scope.name == "unverified_store_profile_diagnostic"
    assert "AnnualSales" in diagnostic_frame
    assert "AnnualSales" in diagnostic_scope.numeric_candidates
    assert "BusinessType" in diagnostic_scope.categorical_features

"""Check paired-error accounting and refuse misleading population joins."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "failure_diagnosis", Path(__file__).with_name("diagnose_frozen_failure.py")
)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def example():
    features = pd.DataFrame({"StoreID": [1, 2, 3], "snapshot": ["2013-10-01"] * 3})
    for feature in diagnosis.FEATURES:
        features[feature] = 1
    rows = []
    for strategy, scores in [
        (diagnosis.BASELINE, [0.1, 0.1, 0.9]),
        (diagnosis.CANDIDATE, [0.9, 0.9, 0.1]),
    ]:
        for store, churn, score in zip([1, 2, 3], [0, 1, 0], scores):
            rows.append(
                dict(
                    StoreID=store,
                    snapshot="2013-10-01",
                    n_orders_3m=1,
                    strategy=strategy,
                    score=score,
                    cutoff=0.5,
                    predicted_churn=int(score >= 0.5),
                    churn=churn,
                )
            )
    return pd.DataFrame(rows), features


def test_paired_change_accounting():
    evaluated, features = example()
    cases = diagnosis.matched_cases(evaluated, features).set_index("StoreID")
    assert cases.change.to_dict() == {
        1: "added_false_alert",
        2: "newly_caught",
        3: "removed_false_alert",
    }
    result = diagnosis.metrics(cases, "predicted_churn_candidate", "score_candidate")
    assert result["recall"] == 1
    assert result["precision"] == 0.5
    assert result["false_positive_rate"] == 0.5


@pytest.mark.parametrize(
    "problem",
    [
        "missing_rule_account",
        "extra_features",
        "duplicate",
        "outcome",
        "decision",
        "activity",
    ],
)
def test_refuse_unmatched_or_inconsistent_evidence(problem):
    evaluated, features = example()
    if problem == "missing_rule_account":
        evaluated = evaluated.iloc[1:]
    elif problem == "extra_features":
        features = pd.concat([features, features.iloc[[0]].assign(StoreID=4)])
    elif problem == "duplicate":
        evaluated = pd.concat([evaluated, evaluated.iloc[[0]]])
    elif problem == "outcome":
        evaluated.loc[0, "churn"] = 1
    elif problem == "decision":
        evaluated.loc[0, "predicted_churn"] = 1
    else:
        features.loc[0, "n_orders_3m"] = 2
    with pytest.raises(ValueError):
        diagnosis.matched_cases(evaluated, features)


def test_prevalence_changes_precision_without_changing_recall_or_fpr():
    high = diagnosis.precision_at_prevalence(0.9, 0.4, 0.1)
    low = diagnosis.precision_at_prevalence(0.9, 0.4, 0.025)
    assert high == pytest.approx(0.2)
    assert low == pytest.approx(0.0545454545)
    assert low < high


def test_saved_cutoff_preserves_float_identity(tmp_path):
    cutoff = 0.21655335464238584
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"cutoff": [cutoff]}).to_csv(path, index=False)
    restored = diagnosis.checked_csv(
        tmp_path, path.name, {"outputs": {path.name: diagnosis.sha(path)}}
    )
    assert restored.cutoff.iloc[0] == cutoff


@pytest.mark.parametrize("problem", ["missing_account", "outcome", "activity"])
def test_development_diagnostics_refuse_inconsistent_joins(problem):
    evaluated, features = example()
    predictions = evaluated[evaluated.strategy.eq(diagnosis.CANDIDATE)].copy()
    predictions["repeat"] = 42
    predictions["outer_fold"] = 0
    development = features.copy()
    development["churn"] = [0, 1, 0]
    if problem == "missing_account":
        development = development.iloc[1:]
    elif problem == "outcome":
        development.loc[0, "churn"] = 1
    else:
        development.loc[0, "n_orders_3m"] = 2
    with pytest.raises(ValueError):
        diagnosis.grouped_diagnostics(predictions, development, "strategy")

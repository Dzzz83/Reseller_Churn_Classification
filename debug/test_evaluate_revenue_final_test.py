"""Source, outcome-population, and overwrite guards for retrospective scoring."""

import pandas as pd
import pytest

import evaluate_revenue_final_test as experiment
from test_prepare_subtotal_totaldue import example_headers


def test_october_feature_source_excludes_outcomes():
    raw = example_headers()
    clean = raw[["SalesOrderID", "StoreID", "OrderDate", "SubTotal"]].copy()
    assert len(experiment.final_headers(clean, raw)) == 5
    raw.loc[0, "OrderDate"] = pd.Timestamp("2013-10-01")
    with pytest.raises(ValueError, match="October or later"):
        experiment.final_headers(clean, raw)


@pytest.mark.parametrize("problem,message", [
    ("missing", "Missing accepted"), ("duplicate", "duplicate raw"),
    ("formula", "formula mismatch"), ("mapping", "Source mismatch"),
])
def test_source_reconciliation_rejects_invalid_data(problem, message):
    raw = example_headers()
    clean = raw[["SalesOrderID", "StoreID", "OrderDate", "SubTotal"]].copy()
    if problem == "missing":
        raw = raw.iloc[1:]
    elif problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif problem == "formula":
        raw.loc[0, "TotalDue"] += 1
    else:
        raw.loc[0, "StoreID"] = 99
    with pytest.raises(ValueError, match=message):
        experiment.final_headers(clean, raw)


def examples():
    labels = pd.DataFrame({"StoreID": range(492), "snapshot": ["2013-10-01"] * 492,
                           "actual_churn": [1] * 46 + [0] * 446})
    cases = labels.drop(columns="actual_churn").copy()
    cases["architecture"], cases["feature_set"] = "random_forest", "subtotal"
    return cases, labels


def test_outcomes_require_exact_population_and_binary_unique_labels():
    cases, labels = examples()
    assert len(experiment.attach_labels(cases, labels)) == 492
    with pytest.raises(ValueError, match="population differs"):
        experiment.attach_labels(cases.iloc[1:], labels)
    with pytest.raises(ValueError, match="Invalid outcome"):
        experiment.attach_labels(cases, pd.concat([labels, labels.iloc[:1]]))
    labels.loc[0, "actual_churn"] = 2
    with pytest.raises(ValueError, match="Invalid outcome"):
        experiment.attach_labels(cases, labels)


def test_existing_evaluation_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    (tmp_path / "results/final_evaluation/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        experiment.run(tmp_path / "absent", tmp_path / "absent", "existing")


def test_scoring_csv_preserves_values_at_histogram_boundaries(tmp_path):
    path = tmp_path / "features.csv"
    value = "0.8207089940237189"
    path.write_text("StoreID,snapshot,share_bikes\n" + "".join(
        f"{store},2013-10-01,{value}\n" for store in range(492)
    ))
    features = experiment.load_final_features(path)
    assert features.share_bikes.eq(float(value)).all()

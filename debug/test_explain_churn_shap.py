"""Probability-scale reconstruction, examples, and native-missing SHAP smoke."""

import numpy as np
import pandas as pd
import pytest

from explain_churn_shap import (
    error_categories, pick_examples, probability_function, verify_reconstruction,
)


def test_reconstruction_rejects_inconsistent_output_scales():
    assert verify_reconstruction([.2], [[.1, -.05]], [.25]) < 1e-12
    with pytest.raises(AssertionError):
        verify_reconstruction([.2], [[.1, -.05]], [.8])


def test_error_categories_include_threshold_equality():
    actual = error_categories([1, 0, 1, 0], [.06, .06, .05, .05], .06)
    assert actual.tolist() == ["caught_churner", "false_alert", "missed_churner", "correct_non_alert"]


def test_walkthrough_is_deterministic_active_false_alert():
    cases = pd.DataFrame({"StoreID": [3, 2, 1, 4], "snapshot": ["2013-04-01"] * 4,
                          "n_orders_3m": [0, 1, 1, 1], "churn_probability": [.99, .8, .8, .9],
                          "error_category": ["false_alert", "false_alert", "false_alert", "caught_churner"]})
    examples = pick_examples(cases)
    assert examples[examples.example.eq("walkthrough")].iloc[0].StoreID == 1
    assert "missed_churner" not in set(examples.example)


def test_probability_permutation_shap_handles_native_missing_values():
    shap = pytest.importorskip("shap")
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(42)
    values = rng.normal(size=(80, 3))
    values[::3, 1] = np.nan
    columns = ["first", "second", "third"]
    data = pd.DataFrame(values, columns=columns)
    target = (values[:, 0] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=5, random_state=42).fit(data, target)
    predict = probability_function(model, columns)
    background = data.iloc[:10].to_numpy()
    cases = data.iloc[10:16].to_numpy()
    explainer = shap.PermutationExplainer(predict,
        shap.maskers.Independent(background, max_samples=10), seed=42)
    result = explainer(cases, max_evals=28, silent=True)
    assert verify_reconstruction(result.base_values, result.values, predict(cases)) < 1e-8
    np.testing.assert_allclose(result.base_values, predict(background).mean(), atol=1e-12, rtol=0)

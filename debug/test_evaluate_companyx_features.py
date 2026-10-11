"""Association edge cases and matched-population checks for the experiment."""

import numpy as np
import pandas as pd
import pytest

from evaluate_companyx_features import (
    NEW_FEATURES, association_tables, benjamini_hochberg, paired_changes,
)
from reseller_churn.config.feature_sets import PRUNED_FEATURES


def test_adjustment_preserves_missing_values_and_original_order():
    actual = benjamini_hochberg([.04, np.nan, .01, .03])
    np.testing.assert_allclose(actual, [.04, np.nan, .03, .04], equal_nan=True)
    assert np.isnan(benjamini_hochberg([np.nan])[0])


def test_associations_report_missing_and_constant_features_by_subgroup():
    data = pd.DataFrame({name: [1., 1., 1., 1.] for name in NEW_FEATURES + PRUNED_FEATURES})
    data["snapshot"] = "2012-10-01"
    data["churn"] = [0, 1, 0, 1]
    data["n_orders_3m"] = [1, 1, 1, 0]
    data["avg_qty_3m"] = [1., np.nan, 2., 3.]
    data["last_salesperson"] = [10, 10, 11, 11]
    data["last_territory"] = [1, 1, 2, 2]
    numeric, categories, redundancy = association_tables(data)
    assert len(numeric) == 2 * len(NEW_FEATURES)
    active = numeric[numeric.subgroup.eq("purchased_in_3m")].set_index("feature")
    assert active.loc["avg_qty_3m", "accounts"] == 3
    assert active.loc["avg_qty_3m", "missing_churners"] == 1
    assert active.loc["avg_qty_3m", "observed"] == 2
    assert pd.isna(active.loc["avg_qty_3m", "pearson_r"])
    assert pd.isna(active.loc["avg_qty_12m", "spearman_r"])
    assert numeric.role.eq("training_period").all()
    assert len(redundancy) == len(NEW_FEATURES) * len(PRUNED_FEATURES)
    assert categories[categories.subgroup.eq("overall")].groupby("field").accounts.sum().eq(4).all()


def example_predictions():
    baseline = pd.DataFrame({
        "fold": ["fold1"] * 5, "StoreID": range(5), "snapshot": ["2013-01-01"] * 5,
        "actual_churn": [1, 1, 0, 0, 0], "n_orders_3m": [1] * 5,
        "churn_probability": [.2, .8, .2, .8, .2], "feature_set": ["baseline"] * 5,
    })
    augmented = baseline.copy()
    augmented["feature_set"] = "discounts"
    augmented["churn_probability"] = [.8, .2, .8, .2, .2]
    return pd.concat([baseline, augmented], ignore_index=True)


def test_paired_changes_classify_all_decision_outcomes():
    selections = {name: {"threshold": .5} for name in ["baseline", "discounts"]}
    paired = paired_changes(example_predictions(), selections, "discounts")
    assert paired.change_type.tolist() == [
        "newly_caught_churn", "newly_missed_churn", "added_false_alert",
        "removed_false_alert", "unchanged",
    ]


@pytest.mark.parametrize("problem", ["missing_case", "changed_label"])
def test_paired_changes_reject_unmatched_populations_and_labels(problem):
    predictions = example_predictions()
    if problem == "missing_case":
        predictions = predictions.drop(index=5)
    else:
        predictions.loc[5, "actual_churn"] = 0
    selections = {name: {"threshold": .5} for name in ["baseline", "discounts"]}
    with pytest.raises(ValueError, match="differ from baseline"):
        paired_changes(predictions, selections, "discounts")

"""Verify frozen-model chronology and saved final-test accounting; no refitting."""

from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_final_testing_evaluation import OUTPUT, ROOT, verify_checkpoint
from reseller_churn.modeling.development import file_sha256


def main():
    protocol = json.loads((OUTPUT / "frozen_protocol.json").read_text())
    checkpoint = json.loads((OUTPUT / "prediction_checkpoint.json").read_text())
    evaluation = json.loads((OUTPUT / "evaluation_manifest.json").read_text())
    access = json.loads((OUTPUT / "outcome_access_started.json").read_text())
    verify_checkpoint(protocol, checkpoint)
    assert datetime.fromisoformat(protocol["frozen_at_utc"]) < datetime.fromisoformat(checkpoint["predictions_saved_at_utc"])
    assert datetime.fromisoformat(checkpoint["predictions_saved_at_utc"]) < datetime.fromisoformat(access["started_at_utc"])
    assert datetime.fromisoformat(access["started_at_utc"]) <= datetime.fromisoformat(evaluation["completed_at_utc"])
    assert evaluation["protected_final_test_labels_accessed"] is True
    assert evaluation["thresholds_changed_after_outcome_access"] is False
    assert evaluation["models_fitted_in_evaluation_stage"] == 0
    for name, expected in evaluation["output_sha256"].items():
        assert file_sha256(OUTPUT / name) == expected, name
    for name, key in [
        ("frozen_protocol.json", "protocol_sha256"),
        ("prediction_checkpoint.json", "checkpoint_sha256"),
        ("outcome_access_started.json", "outcome_access_marker_sha256"),
    ]:
        assert file_sha256(OUTPUT / name) == evaluation[key]
    cases = pd.read_csv(OUTPUT / "final_test_predictions.csv")
    unlabeled = pd.read_csv(OUTPUT / "unlabeled_predictions.csv")
    assert len(cases) == len(unlabeled) == evaluation["test_rows"] == 492
    assert cases.actual_churn.sum() == evaluation["test_churners"] == 46
    assert not cases.StoreID.duplicated().any()
    assert cases.snapshot.eq("2013-10-01").all()
    pd.testing.assert_frame_equal(cases[unlabeled.columns], unlabeled)
    individual = pd.read_csv(OUTPUT / "individual_seed_scores.csv")
    np.testing.assert_array_equal(individual.StoreID, unlabeled.StoreID)
    np.testing.assert_allclose(
        individual.drop(columns="StoreID").mean(axis=1), unlabeled.churn_probability, atol=1e-14,
    )
    metrics = pd.read_csv(OUTPUT / "final_test_metrics.csv")
    for _, row in metrics.iterrows():
        group = cases
        if row.subgroup == "purchased_in_3m":
            group = cases[cases.n_orders_3m.gt(0)]
        elif row.subgroup == "no_purchase_in_3m":
            group = cases[cases.n_orders_3m.eq(0)]
        else:
            assert row.subgroup == "overall"
        rule = next(rule for rule in protocol["rules"] if rule["minimum_development_precision"] == row.minimum_development_precision)
        assert rule["role"] == row.role
        assert rule["threshold"] == row.threshold
        positive, alert = group.actual_churn.eq(1), group.churn_probability.ge(row.threshold)
        tp, fp = int((positive & alert).sum()), int((~positive & alert).sum())
        fn, tn = int((positive & ~alert).sum()), int((~positive & ~alert).sum())
        assert [tp, fp, fn, tn] == [row.tp, row.fp, row.fn, row.tn]
        assert len(group) == row.rows
        assert positive.sum() == row.churners
        np.testing.assert_allclose(row.recall, tp / max(1, tp + fn))
        np.testing.assert_allclose(row.precision, tp / max(1, tp + fp))
        np.testing.assert_allclose(row.f1, 2 * tp / max(1, 2 * tp + fp + fn))
        np.testing.assert_allclose(row.average_precision, average_precision_score(positive, group.churn_probability))
        np.testing.assert_allclose(row.roc_auc, roc_auc_score(positive, group.churn_probability))
    assert len(metrics) == 12
    assert metrics[metrics.role.eq("primary") & metrics.subgroup.eq("overall")].shape[0] == 1
    print("Verified frozen protocol and prediction-before-label chronology, five-model seed averaging, all 12 final metric rows, unchanged alert rules, and artifact hashes.")


if __name__ == "__main__":
    main()

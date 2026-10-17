"""Development boundaries, adverse periods and frozen-cutoff transfer reporting."""

import numpy as np
import pandas as pd
import pytest

from comparisons.development_checks import (
    validate_development_cases,
    validate_development_split,
    validate_feature_times,
)
from comparisons.development_reporting import (
    cutoff_transfer_tables,
    paired_operating_tables,
)
from comparisons.review_development_study import review
from reseller_churn.artifacts import write_json


def cases(snapshot="2012-07-01", stores=(1, 2, 3, 4)):
    return pd.DataFrame(
        dict(
            StoreID=stores,
            snapshot=pd.Timestamp(snapshot),
            churn=[1, 0, 1, 0],
            n_orders_3m=[1, 1, 0, 0],
            x=[1.0, 2.0, np.nan, 4.0],
        )
    )


@pytest.mark.parametrize(
    "snapshot", ["2013-10-01", "2013-07-01", "2014-01-01", "2012-07-01 00:00:01"]
)
def test_unapproved_dates_rejected(snapshot):
    with pytest.raises(ValueError, match="approved development"):
        validate_development_cases(cases(snapshot))


def test_predictors_exclude_identity_and_labels_but_preserve_native_missingness():
    frame = cases()
    validate_development_cases(frame, ["x"])
    for column in ["StoreID", "snapshot", "churn"]:
        with pytest.raises(ValueError, match="cannot be predictors"):
            validate_development_cases(frame, [column])
    with pytest.raises(ValueError, match="outcome-derived"):
        validate_development_cases(frame.assign(actual_churn=0))
    with pytest.raises(ValueError, match="Infinite"):
        validate_development_cases(frame.assign(x=np.inf), ["x"])


def test_temporal_grouped_maturity_boundary_and_reseller_isolation():
    train = cases()
    valid = cases("2013-01-01", (5, 6, 7, 8))
    validate_development_split(train, valid, design="temporal_grouped")
    with pytest.raises(ValueError, match="not mature"):
        validate_development_split(
            cases("2012-10-01"), valid, design="temporal_grouped"
        )
    with pytest.raises(ValueError, match="Reseller leakage"):
        validate_development_split(train, cases("2013-01-01"))
    # Grouped estimation deliberately makes no forecasting claim.
    validate_development_split(cases("2013-04-01"), valid, design="grouped")


def test_feature_end_is_exclusive():
    validate_feature_times(pd.Series(["2012-06-30"]), "2012-07-01")
    for day in ["2012-07-01", "2012-07-02"]:
        with pytest.raises(ValueError, match="end is exclusive"):
            validate_feature_times(pd.Series([day]), "2012-07-01")


def predictions():
    rows = []
    for repeat in [1, 2, 3]:
        for arm in ["baseline", "candidate"]:
            for day in ["2012-07-01", "2013-04-01"]:
                # Overall improves, but April continuing buyer gets a new alert.
                score = (
                    [0.8, 0.8, 0.8, 0.8]
                    if day == "2012-07-01"
                    else [0.8, 0.1, 0.8, 0.1]
                )
                if arm == "candidate":
                    score = (
                        [0.8, 0.1, 0.8, 0.1]
                        if day == "2012-07-01"
                        else [0.8, 0.5, 0.8, 0.1]
                    )
                rows.append(
                    cases(day).assign(
                        strategy=arm,
                        repeat=repeat,
                        outer_fold=[0, 0, 1, 1],
                        score=score,
                        cutoff=0.5,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def test_average_gain_does_not_hide_period_harm_and_cutoff_equality():
    raw, paired, worst = paired_operating_tables(predictions(), "candidate")
    assert (
        paired[paired.slice.eq("overall") & paired.scope.eq("repeat_pooled")]
        .false_alert_reduction.eq(1)
        .all()
    )
    april = paired[
        paired.slice.eq("snapshot")
        & paired.group.eq("2013-04-01")
        & paired.scope.eq("repeat_pooled")
    ]
    assert april.more_false_alerts_than_baseline.all()
    assert april.false_alert_reduction.eq(-1).all()
    assert worst.iloc[0].false_alert_reduction == -1
    assert {"snapshot_activity", "snapshot", "activity", "overall"} == set(raw.slice)
    assert "accepted" not in paired and "accepted" not in worst


@pytest.mark.parametrize("change", ["fold", "population", "label", "test", "score"])
def test_unmatched_or_invalid_predictions_rejected(change):
    frame = predictions()
    candidate = frame.index[frame.strategy.eq("candidate")][0]
    if change == "fold":
        frame.loc[candidate, "outer_fold"] = 9
    elif change == "population":
        frame = frame.drop(candidate)
    elif change == "label":
        frame.loc[candidate, "churn"] = 0
    elif change == "test":
        frame.loc[candidate, "snapshot"] = pd.Timestamp("2013-10-01")
    else:
        frame.loc[candidate, "score"] = np.inf
    with pytest.raises((ValueError, AssertionError)):
        paired_operating_tables(frame, "candidate")


def test_empty_and_single_class_slices_have_undefined_rates():
    frame = predictions()
    frame.n_orders_3m = 1
    _, paired, _ = paired_operating_tables(frame, "candidate")
    empty = paired[paired.group.eq("no_recent_purchase")]
    assert empty.rows_baseline.eq(0).all()
    assert not empty.recall_comparison_defined.any()
    assert empty.recall_difference.isna().all()


def test_transfer_uses_frozen_cutoff_and_records_actual_training_sizes():
    development = pd.concat([cases(), cases(stores=(5, 6, 7, 8))], ignore_index=True)
    inner = development[development.StoreID.le(4)].assign(
        candidate="candidate", inner_fold=[0, 0, 1, 1], score=[0.8, 0.5, 0.2, 0.1]
    )
    outer = development[development.StoreID.gt(4)].assign(
        strategy="candidate",
        repeat=1,
        outer_fold=0,
        score=[0.9, 0.5, 0.8, 0.1],
        cutoff=0.5,
    )
    untouched = inner.copy(deep=True)
    table = cutoff_transfer_tables(inner, outer, development)
    whole = table[table.stage.eq("inner_oof") & table.activity.eq("overall")].iloc[0]
    assert whole.tp == 1 and whole.fp == 1
    assert whole.training_rows_min == whole.training_rows_max == 2
    full = table[
        table.stage.eq("outer_validation") & table.activity.eq("overall")
    ].iloc[0]
    assert full.training_rows_min == full.training_resellers_min == 4
    assert table.cutoff.eq(0.5).all()
    pd.testing.assert_frame_equal(inner, untouched)
    with pytest.raises(ValueError, match="leakage"):
        cutoff_transfer_tables(inner, outer.assign(StoreID=[1, 6, 7, 8]), development)


def test_review_rejects_non_development_role_before_reading_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", {"stage": "evaluated"})
    with pytest.raises(ValueError, match="fixed_numeric_feature"):
        review(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_saved_study_review_is_no_fit_complete_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    from comparisons import grouped_precision_recall as study
    from reseller_churn.artifacts import write_manifest

    def forbidden(*args, **kwargs):
        raise AssertionError("Review must never fit or select a cutoff")

    monkeypatch.setattr(study, "fit_scores", forbidden)
    monkeypatch.setattr(study, "select_threshold", forbidden)
    source = tmp_path / "saved"
    source.mkdir()
    data = pd.concat([cases(), cases(stores=(5, 6, 7, 8))], ignore_index=True)
    data.to_csv(source / "development_input.csv", index=False)
    parts, splits = [], []
    for fold in [0, 1]:
        ti = list(range(4 * fold, 4 * fold + 4))
        vi = [i for i in range(8) if i not in ti]
        train, valid = data.iloc[ti], data.iloc[vi]
        splits.append(
            dict(
                repeat=1,
                outer_fold=fold,
                train_rows=ti,
                validation_rows=vi,
                train_store_ids=train.StoreID.tolist(),
                validation_store_ids=valid.StoreID.tolist(),
            )
        )
        folder = source / f"repeat_1/fold_{fold}"
        folder.mkdir(parents=True)
        inner, outer = [], []
        for arm in ["baseline", "candidate"]:
            inner.append(
                train[["StoreID", "snapshot", "churn"]].assign(
                    candidate=arm, inner_fold=[0, 0, 1, 1], score=[0.8, 0.3, 0.2, 0.1]
                )
            )
            outer.append(
                valid.assign(
                    strategy=arm,
                    repeat=1,
                    outer_fold=fold,
                    score=[0.8, 0.3, 0.2, 0.1],
                    cutoff=0.5,
                )
            )
        pd.concat(inner).to_csv(folder / "inner_scores.csv", index=False)
        part = pd.concat(outer)
        part.to_csv(folder / "predictions.csv", index=False)
        parts.append(part)
        write_manifest(folder, {})
    pd.concat(parts).to_csv(source / "validation_predictions.csv", index=False)
    write_json(
        source / "protocol.json",
        {
            "arms": [
                {"id": arm, "features": ["x"]} for arm in ["baseline", "candidate"]
            ],
            "splits": splits,
        },
    )
    write_manifest(source, {"stage": "fixed_numeric_feature"})
    output = tmp_path / "review"
    review(source, output, canonical=False)
    assert (output / "report.md").exists()
    report = pd.read_csv(output / "cutoff_transfer.csv")
    assert report.cutoff.eq(0.5).all()
    assert set(report.stage) == {"inner_oof", "inner_oof_fold", "outer_validation"}
    assert (
        pd.read_csv(output / "paired_operating_metrics.csv")
        .false_alert_reduction.eq(0)
        .all()
    )
    with pytest.raises(FileExistsError):
        review(source, output, canonical=False)


def test_frozen_period_limits_reject_harm_even_when_total_fp_improves():
    from comparisons.development_reporting import assess_period_safeguards

    _, paired, _ = paired_operating_tables(predictions(), "candidate")
    passed, checks = assess_period_safeguards(
        paired, max_recall_loss=0.05, max_fp_increase=0
    )
    assert not passed
    assert checks[checks.group.eq("2013-04-01")].status.eq("failed").all()
    assert assess_period_safeguards(paired, max_recall_loss=0.05, max_fp_increase=1)[0]
    raw = predictions()
    raw.n_orders_3m = 1
    _, paired, _ = paired_operating_tables(raw, "candidate")
    passed, checks = assess_period_safeguards(
        paired, max_recall_loss=0.05, max_fp_increase=1
    )
    assert not passed
    assert (
        checks[checks.group.eq("no_recent_purchase")].status.eq("not_evaluable").all()
    )
    with pytest.raises(ValueError, match="nonnegative count"):
        assess_period_safeguards(paired, max_recall_loss=0.05, max_fp_increase=-1)


def test_invalid_period_limits_rejected_before_fitting_or_source_access(tmp_path):
    from comparisons.fixed_feature_comparison import run_fixed_hgb

    with pytest.raises(ValueError, match="nonnegative count"):
        run_fixed_hgb(
            cases(),
            None,
            tmp_path / "missing",
            tmp_path / "output",
            {"features": ["x"]},
            tmp_path / "missing.md",
            0.025,
            {},
            period_limits={"max_recall_loss": 0.05, "max_fp_increase": -1},
        )
    assert not (tmp_path / "output").exists()


def test_shared_assessment_applies_explicit_period_gate(tmp_path):
    from comparisons.grouped_precision_recall import FEATURES
    from comparisons.matched_evaluation import evaluate_predictions

    predictions_frame = predictions()
    features = predictions_frame.drop_duplicates(["StoreID", "snapshot"])[
        ["StoreID", "snapshot", "churn", "n_orders_3m"]
    ].copy()
    for column in FEATURES:
        if column not in features:
            features[column] = 1.0
    profiles = features[["StoreID", "snapshot"]].assign(
        n_orders_12m=4, observed_history_span_days=200
    )
    result = evaluate_predictions(
        features,
        predictions_frame,
        profiles,
        tmp_path,
        "candidate",
        0.025,
        period_limits={"max_recall_loss": 0.05, "max_fp_increase": 0},
    )
    assert not result["checks"]["frozen_period_activity_limits"]
    assert not result["accepted"]
    assert (tmp_path / "period_safeguards.csv").exists()
    assert (tmp_path / "worst_slices.csv").exists()

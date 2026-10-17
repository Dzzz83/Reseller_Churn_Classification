"""Matched development operating tables and descriptive cutoff-transfer checks."""

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.analysis import summarize_group
from comparisons.development_checks import KEYS, validate_development_cases

COLUMNS = [
    "rows",
    "resellers",
    "churners",
    "non_churners",
    "tp",
    "fp",
    "fn",
    "tn",
    "recall",
    "precision",
    "false_positive_rate",
    "alert_count",
    "alert_rate",
    "small_sample",
]


def validate_predictions(predictions, candidate_id):
    required = {
        *KEYS,
        "churn",
        "n_orders_3m",
        "score",
        "cutoff",
        "repeat",
        "outer_fold",
        "strategy",
    }
    if required - set(predictions) or predictions.empty:
        raise ValueError("Missing development prediction columns")
    if candidate_id == "baseline" or set(predictions.strategy) != {
        "baseline",
        candidate_id,
    }:
        raise ValueError("Expected original baseline and exactly one candidate")
    if predictions[["repeat", "outer_fold"]].isna().any().any():
        raise ValueError("Missing split identity")
    for name in ("score", "cutoff"):
        if (
            not np.isfinite(predictions[name]).all()
            or not predictions[name].between(0, 1).all()
        ):
            raise ValueError("Invalid development scores or cutoffs")
    if (
        not np.isfinite(predictions.n_orders_3m).all()
        or not (
            predictions.n_orders_3m.ge(0)
            & predictions.n_orders_3m.eq(np.floor(predictions.n_orders_3m))
        ).all()
    ):
        raise ValueError("Invalid pre-snapshot order counts")
    identities = ["strategy", "repeat", *KEYS]
    if predictions.duplicated(identities).any():
        raise ValueError("Duplicate development predictions")
    reference = None
    for _, rows in predictions.groupby(["repeat", "strategy"]):
        validate_development_cases(rows)
        content = (
            rows[KEYS + ["churn", "n_orders_3m"]]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        if reference is None:
            reference = content
        else:
            pd.testing.assert_frame_equal(content, reference)
    a = predictions[predictions.strategy.eq("baseline")]
    b = predictions[predictions.strategy.eq(candidate_id)]
    paired = a.merge(
        b,
        on=[*KEYS, "repeat"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if not paired.outer_fold_baseline.eq(paired.outer_fold_candidate).all():
        raise ValueError("Baseline/candidate validation folds differ")
    if (
        not predictions.groupby(["strategy", "repeat", "StoreID"])
        .outer_fold.nunique()
        .eq(1)
        .all()
    ):
        raise ValueError("Reseller appears in multiple outer validation folds")
    if (
        not predictions.groupby(["strategy", "repeat", "outer_fold"])
        .cutoff.nunique()
        .eq(1)
        .all()
    ):
        raise ValueError("Expected one training-selected cutoff per fitted outer model")


def operating_summary(rows, cutoff=None):
    frame = rows.assign(
        predicted_churn=rows.score.ge(rows.cutoff if cutoff is None else cutoff).astype(
            int
        )
    )
    summary = summarize_group(frame)
    return {name: summary[name] for name in COLUMNS}


def slices(frame):
    """Include empty activity cells explicitly, with undefined rates preserved."""
    yield "overall", "all", frame
    for name, recent in [("recent_purchase", True), ("no_recent_purchase", False)]:
        yield "activity", name, frame[frame.n_orders_3m.gt(0).eq(recent)]
    for snapshot, part in frame.groupby("snapshot"):
        day = pd.Timestamp(snapshot).strftime("%Y-%m-%d")
        yield "snapshot", day, part
        for name, recent in [("recent_purchase", True), ("no_recent_purchase", False)]:
            yield (
                "snapshot_activity",
                f"{day}/{name}",
                part[part.n_orders_3m.gt(0).eq(recent)],
            )


def paired_operating_tables(predictions, candidate_id):
    validate_predictions(predictions, candidate_id)
    metrics = []
    for scope, keys in [
        ("repeat_pooled", ["strategy", "repeat"]),
        ("outer_fold", ["strategy", "repeat", "outer_fold"]),
    ]:
        for identity, frame in predictions.groupby(keys):
            metadata = {"outer_fold": -1, **dict(zip(keys, identity)), "scope": scope}
            for dimension, group, rows in slices(frame):
                metrics.append(
                    {
                        **metadata,
                        "slice": dimension,
                        "group": group,
                        **operating_summary(rows),
                    }
                )
    metrics = pd.DataFrame(metrics)
    keys = ["scope", "repeat", "outer_fold", "slice", "group"]
    paired = metrics[metrics.strategy.eq("baseline")].merge(
        metrics[metrics.strategy.eq(candidate_id)],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["recall_difference"] = paired.recall_candidate - paired.recall_baseline
    paired["false_alert_reduction"] = paired.fp_baseline - paired.fp_candidate
    paired["recall_lower_than_baseline"] = paired.recall_difference.lt(0)
    paired["more_false_alerts_than_baseline"] = paired.false_alert_reduction.lt(0)
    paired["recall_comparison_defined"] = paired.recall_difference.notna()
    # A sorted diagnostic, never a newly imposed acceptance rule or winning slice.
    worst = paired[
        paired.scope.eq("repeat_pooled") & paired.slice.ne("overall")
    ].sort_values(
        ["false_alert_reduction", "recall_difference", "repeat", "group"],
        na_position="first",
    )
    return metrics, paired, worst


def cutoff_transfer_tables(inner, outer, development):
    """Use saved outer cutoffs on inner OOF and outer scores; never select one."""
    validate_development_cases(development)
    outer = outer.copy()
    inner = inner.copy()
    for frame in (development, outer, inner):
        # Local callers supply parsed dates; copies below prevent caller mutation.
        if not pd.api.types.is_datetime64_any_dtype(frame.snapshot):
            raise ValueError("Parse snapshot dates before cutoff-transfer reporting")
    if outer[["repeat", "outer_fold"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Cutoff transfer expects exactly one outer split")
    if set(inner.candidate) != set(outer.strategy):
        raise ValueError("Inner/outer strategies differ")
    if set(inner.StoreID) & set(outer.StoreID):
        raise ValueError("Inner/outer reseller leakage")
    rows = []
    assignment = None
    for arm, outside in outer.groupby("strategy"):
        inside = inner[inner.candidate.eq(arm)].copy()
        validate_development_cases(inside)
        validate_development_cases(outside)
        if not np.isfinite(inside.score).all() or not inside.score.between(0, 1).all():
            raise ValueError("Invalid inner OOF scores")
        current_assignment = (
            inside[KEYS + ["inner_fold"]].sort_values(KEYS).reset_index(drop=True)
        )
        if assignment is None:
            assignment = current_assignment
        else:
            pd.testing.assert_frame_equal(current_assignment, assignment)
        if (
            inside.inner_fold.isna().any()
            or inside.inner_fold.nunique() < 2
            or not inside.groupby("StoreID").inner_fold.nunique().eq(1).all()
        ):
            raise ValueError("Invalid inner reseller assignments")
        truth = development[KEYS + ["churn", "n_orders_3m"]]
        matched = inside.merge(
            truth, on=KEYS, how="left", suffixes=("", "_source"), validate="one_to_one"
        )
        if (
            matched.churn_source.isna().any()
            or not matched.churn.eq(matched.churn_source).all()
        ):
            raise ValueError("Inner outcomes do not match development cases")
        if "n_orders_3m_source" in matched:
            if not matched.n_orders_3m.eq(matched.n_orders_3m_source).all():
                raise ValueError("Inner activity counts differ from development")
            matched = matched.drop(columns="n_orders_3m_source")
        inside = matched.drop(columns="churn_source")
        cutoffs = outside.cutoff.unique()
        if len(cutoffs) != 1:
            raise ValueError("Outer model must have one frozen cutoff")
        cutoff = cutoffs[0]
        sizes = [
            inside[inside.inner_fold.ne(fold)]
            for fold in sorted(inside.inner_fold.unique())
        ]
        populations = [
            (
                "inner_oof",
                -1,
                inside,
                min(len(x) for x in sizes),
                max(len(x) for x in sizes),
                min(x.StoreID.nunique() for x in sizes),
                max(x.StoreID.nunique() for x in sizes),
            ),
            (
                "outer_validation",
                -1,
                outside,
                len(inside),
                len(inside),
                inside.StoreID.nunique(),
                inside.StoreID.nunique(),
            ),
        ]
        for fold, part in inside.groupby("inner_fold"):
            fitting = inside[inside.inner_fold.ne(fold)]
            populations.append(
                (
                    "inner_oof_fold",
                    int(fold),
                    part,
                    len(fitting),
                    len(fitting),
                    fitting.StoreID.nunique(),
                    fitting.StoreID.nunique(),
                )
            )
        for stage, fold, population, lo, hi, stores_lo, stores_hi in populations:
            for activity, subset in [
                ("overall", population),
                ("recent_purchase", population[population.n_orders_3m.gt(0)]),
                ("no_recent_purchase", population[population.n_orders_3m.eq(0)]),
            ]:
                rows.append(
                    dict(
                        strategy=arm,
                        repeat=outside.repeat.iloc[0],
                        outer_fold=outside.outer_fold.iloc[0],
                        stage=stage,
                        inner_fold=fold,
                        activity=activity,
                        cutoff=cutoff,
                        training_rows_min=lo,
                        training_rows_max=hi,
                        training_resellers_min=stores_lo,
                        training_resellers_max=stores_hi,
                        **operating_summary(subset, cutoff),
                    )
                )
    return pd.DataFrame(rows)


def assess_period_safeguards(paired, *, max_recall_loss, max_fp_increase):
    """Apply explicitly frozen limits per repeat, never choose limits from results.

    Global recall/FP, practical and uncertainty acceptance still apply separately.
    Undefined recall cannot pass; it is reported as not evaluable, not zero loss.
    """
    validate_period_limits(
        max_recall_loss=max_recall_loss, max_fp_increase=max_fp_increase
    )
    rows = paired[
        paired.scope.eq("repeat_pooled") & paired.slice.isin(["snapshot", "activity"])
    ].copy()
    if rows.empty or set(rows.slice) != {"snapshot", "activity"}:
        raise ValueError("Snapshot and activity evidence are required")
    rows["recall_limit_passed"] = (
        rows.recall_comparison_defined & rows.recall_difference.ge(-max_recall_loss)
    )
    rows["fp_limit_passed"] = rows.false_alert_reduction.ge(-max_fp_increase)
    rows["status"] = np.where(
        ~rows.recall_comparison_defined,
        "not_evaluable",
        np.where(rows.recall_limit_passed & rows.fp_limit_passed, "passed", "failed"),
    )
    return bool(rows.status.eq("passed").all()), rows


def validate_period_limits(*, max_recall_loss, max_fp_increase):
    """Reject invalid frozen settings before fitting or writing an assessment."""
    if not np.isfinite(max_recall_loss) or not 0 <= max_recall_loss <= 1:
        raise ValueError("Recall-loss limit must be a fraction in [0, 1]")
    if (
        not np.isfinite(max_fp_increase)
        or max_fp_increase < 0
        or int(max_fp_increase) != max_fp_increase
    ):
        raise ValueError("False-alert increase limit must be a nonnegative count")

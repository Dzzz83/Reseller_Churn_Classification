"""Pure transformations of held-out baseline predictions and historical profiles."""

from pathlib import Path

import numpy as np
import pandas as pd

from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.data.prediction_window import PredictionWindow
from reseller_churn.evaluation.metrics import calculate_ranking_metrics

from .definitions import KEYS, POLICIES, RANK_BANDS, SLICES
from .inputs import require


def historical_profiles(features: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Describe only accepted purchases within each snapshot's history window."""
    profiles = []
    for snapshot, keys in features.groupby("snapshot"):
        history = PredictionWindow.from_date(snapshot).select_history(orders)
        profile = (
            history.groupby("StoreID")
            .agg(
                first_observed_order=("OrderDate", "min"),
                last_observed_order=("OrderDate", "max"),
                n_orders_12m=("SalesOrderID", "nunique"),
            )
            .reindex(keys.StoreID)
            .reset_index()
        )
        require(
            profile.n_orders_12m.notna().all(),
            "Missing historical purchases for eligible reseller",
        )
        profile["observed_history_span_days"] = (
            profile.last_observed_order - profile.first_observed_order
        ).dt.days
        profile["snapshot"] = snapshot
        profiles.append(profile)
    return pd.concat(profiles, ignore_index=True)


def build_cases(
    features: pd.DataFrame, predictions: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    descriptive = features[[*KEYS, *FEATURES]].merge(
        profiles, on=KEYS, validate="one_to_one"
    )
    descriptive["activity"] = np.where(
        descriptive.n_orders_3m.gt(0), "recent_purchase", "no_recent_purchase"
    )
    descriptive["annual_orders"] = pd.cut(
        descriptive.n_orders_12m, [0, 1, 2, np.inf], labels=SLICES["annual_orders"]
    )
    descriptive["history_span"] = pd.cut(
        descriptive.observed_history_span_days,
        [-1, 90, 180, np.inf],
        labels=SLICES["history_span"],
    )
    for name in FEATURES[-3:]:
        descriptive[f"{name}_availability"] = np.where(
            descriptive[name].isna(), "missing", "available"
        )
    descriptive["snapshot_slice"] = descriptive.snapshot.dt.strftime("%Y-%m-%d")
    cases = predictions.drop(columns="n_orders_3m").merge(
        descriptive, on=KEYS, validate="many_to_one"
    )
    cases["policy"] = cases.strategy.map(POLICIES)
    grouping = cases.groupby(["policy", "repeat", "outer_fold"])
    rank = grouping.score.rank(ascending=False, method="average")
    cases["rank_percentile"] = (rank - 0.5) / grouping.score.transform("size")
    cases["rank_band"] = pd.cut(
        cases.rank_percentile,
        [0, 0.10, 0.25, 0.50, 0.75, 1],
        labels=RANK_BANDS,
        include_lowest=True,
    )
    cases["predicted_churn"] = cases.score.ge(cases.cutoff).astype(int)
    cases["boundary_percentile"] = grouping.predicted_churn.transform("mean")
    cases["rank_distance_to_boundary"] = (
        cases.rank_percentile - cases.boundary_percentile
    )
    cases["near_boundary"] = cases.rank_distance_to_boundary.abs().le(0.05)
    positive, alert = cases.churn.eq(1), cases.predicted_churn.eq(1)
    cases["result_type"] = np.select(
        [positive & alert, ~positive & alert, positive & ~alert],
        ["caught_churn", "false_alert", "missed_churn"],
        default="correct_non_churn",
    )
    return cases


def summarize_group(frame: pd.DataFrame) -> dict:
    """Use explicit undefined rates for empty or single-class slices."""
    rows = len(frame)
    positives = int(frame.churn.sum())
    negatives = rows - positives
    if not rows:
        return dict(
            rows=0,
            resellers=0,
            churners=0,
            non_churners=0,
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            recall=None,
            precision=None,
            f1=None,
            false_positive_rate=None,
            alert_count=0,
            alert_rate=None,
            prevalence=None,
            average_precision=None,
            roc_auc=None,
            small_sample=True,
        )
    # Cutoffs vary by outer fold in pooled reports, so summarize saved decisions.
    positive, alert = frame.churn.eq(1), frame.predicted_churn.eq(1)
    tp, fp, fn, tn = [
        int(mask.sum())
        for mask in (
            positive & alert,
            ~positive & alert,
            positive & ~alert,
            ~positive & ~alert,
        )
    ]
    ranking = calculate_ranking_metrics(frame.churn, frame.score)
    return dict(
        rows=rows,
        resellers=frame.StoreID.nunique(),
        churners=positives,
        non_churners=negatives,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=tp / positives if positives else None,
        precision=tp / (tp + fp) if tp + fp else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        false_positive_rate=fp / negatives if negatives else None,
        alert_count=tp + fp,
        alert_rate=(tp + fp) / rows,
        prevalence=positives / rows,
        average_precision=ranking["average_precision"],
        roc_auc=ranking["roc_auc"],
        small_sample=rows < 30 or positives < 10 or negatives < 10,
    )


def metric_tables(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bands = [], []
    for scope, columns in [
        ("repeat_pooled", ["policy", "repeat"]),
        ("outer_fold", ["policy", "repeat", "outer_fold"]),
    ]:
        for identity, frame in cases.groupby(columns):
            metadata = dict(zip(columns, identity))
            metadata["scope"] = scope
            for dimension, values in {"overall": ("all",), **SLICES}.items():
                for value in values:
                    subset = (
                        frame
                        if dimension == "overall"
                        else frame[frame[dimension].eq(value)]
                    )
                    rows.append(
                        {
                            **metadata,
                            "slice": dimension,
                            "group": value,
                            **summarize_group(subset),
                        }
                    )
            for band in RANK_BANDS:
                bands.append(
                    {
                        **metadata,
                        "rank_band": band,
                        **summarize_group(frame[frame.rank_band.eq(band)]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(bands)


def rank_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """All unique score boundaries; tied accounts always enter together."""
    groups = (
        frame.groupby("score").churn.agg(["sum", "size"]).sort_index(ascending=False)
    )
    tp = groups["sum"].cumsum()
    fp = (groups["size"] - groups["sum"]).cumsum()
    positives = int(frame.churn.sum())
    return pd.DataFrame(
        {
            "score_boundary": groups.index,
            "tp": tp.to_numpy(),
            "fp": fp.to_numpy(),
            "recall": tp.to_numpy() / positives if positives else np.nan,
            "precision": (tp / (tp + fp)).to_numpy(),
        }
    )


def ranking_tables(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves, burdens = [], []
    # Ranking is identical for both policies; report it once.
    baseline = cases[cases.policy.eq("inner_selected")]
    for scope, columns in [
        ("repeat_pooled_mixed_models", ["repeat"]),
        ("outer_fold", ["repeat", "outer_fold"]),
    ]:
        for identity, frame in baseline.groupby(columns):
            metadata = {**dict(zip(columns, identity)), "scope": scope}
            for subgroup, subset in [
                ("overall", frame),
                ("recent_purchase", frame[frame.n_orders_3m.gt(0)]),
            ]:
                if subset.empty:
                    continue
                curve = rank_curve(subset)
                for name, value in {**metadata, "subgroup": subgroup}.items():
                    curve[name] = value
                curves.append(curve)
                for target in (0.90, 0.95, 1.0):
                    eligible = curve[curve.recall.ge(target)]
                    point = eligible.iloc[0] if not eligible.empty else None
                    burdens.append(
                        {
                            **metadata,
                            "subgroup": subgroup,
                            "target_recall": target,
                            "churners": int(subset.churn.sum()),
                            "rows": len(subset),
                            "achieved_recall": point.recall
                            if point is not None
                            else None,
                            "tp": int(point.tp) if point is not None else None,
                            "fp": int(point.fp) if point is not None else None,
                            "precision": point.precision if point is not None else None,
                            "diagnostic_only": True,
                        }
                    )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(burdens)


def stability_table(cases: pd.DataFrame) -> pd.DataFrame:
    table = (
        cases.groupby(["policy", *KEYS])
        .agg(
            churn=("churn", "first"),
            repetitions=("repeat", "nunique"),
            alerts=("predicted_churn", "sum"),
            rank_mean=("rank_percentile", "mean"),
            rank_min=("rank_percentile", "min"),
            rank_max=("rank_percentile", "max"),
            activity=("activity", "first"),
            annual_orders=("annual_orders", "first"),
            history_span=("history_span", "first"),
        )
        .reset_index()
    )
    table["false_alert_repetitions"] = np.where(table.churn.eq(0), table.alerts, 0)
    table["missed_churn_repetitions"] = np.where(
        table.churn.eq(1), table.repetitions - table.alerts, 0
    )
    errors = table.false_alert_repetitions + table.missed_churn_repetitions
    table["error_pattern"] = np.select(
        [errors.eq(table.repetitions), errors.eq(2), errors.eq(1)],
        ["consistent_error", "recurring_error", "occasional_error"],
        default="no_error",
    )
    table["rank_range"] = table.rank_max - table.rank_min
    table["rank_unstable"] = table.rank_range.ge(0.20)
    table["decision_unstable"] = table.alerts.gt(0) & table.alerts.lt(table.repetitions)
    agreement = cases.pivot(
        index=[*KEYS, "repeat"], columns="policy", values="predicted_churn"
    )
    agreement["policies_agree"] = agreement.fixed_006.eq(agreement.inner_selected)
    agreement = (
        agreement.groupby(KEYS)
        .policies_agree.sum()
        .rename("policy_agreement_repetitions")
    )
    return table.merge(agreement, on=KEYS, validate="many_to_one")


def verify_saved_metrics(metrics: pd.DataFrame, source: Path) -> int:
    checked = 0
    mapping = {
        "all": "overall",
        "recent_purchase": "active",
        "no_recent_purchase": "inactive",
    }
    for scope, filename, keys in [
        ("repeat_pooled", "repeat_metrics.csv", ["repeat", "strategy", "subgroup"]),
        (
            "outer_fold",
            "fold_metrics.csv",
            ["repeat", "outer_fold", "strategy", "subgroup"],
        ),
    ]:
        actual = metrics[
            (metrics.scope == scope) & metrics.slice.isin(["overall", "activity"])
        ].copy()
        actual["subgroup"] = actual.group.map(mapping)
        actual["strategy"] = actual.policy.map({v: k for k, v in POLICIES.items()})
        saved = pd.read_csv(source / filename, float_precision="round_trip")
        saved = saved[saved.strategy.isin(POLICIES)]
        pair = actual.merge(
            saved,
            on=keys,
            how="outer",
            validate="one_to_one",
            suffixes=("", "_saved"),
            indicator=True,
        )
        require(pair["_merge"].eq("both").all(), "Saved metric populations differ")
        for name in (
            "rows",
            "churners",
            "tp",
            "fp",
            "fn",
            "tn",
            "recall",
            "precision",
            "average_precision",
        ):
            np.testing.assert_allclose(
                pair[name], pair[f"{name}_saved"], rtol=0, atol=1e-12, equal_nan=True
            )
        checked += len(pair)
    return checked

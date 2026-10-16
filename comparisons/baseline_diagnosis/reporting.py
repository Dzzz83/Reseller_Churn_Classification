"""Plots and a human-readable interpretation of the saved-prediction diagnosis."""

from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import REPEATS


def plot_diagnostics(
    cases: pd.DataFrame, curves: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = cases[cases.policy.eq("inner_selected")]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for column, repeat in enumerate(REPEATS):
        for row, (name, active_only) in enumerate(
            [("All accounts", False), ("Recently purchasing", True)]
        ):
            frame = baseline[baseline.repeat.eq(repeat)]
            if active_only:
                frame = frame[frame.n_orders_3m.gt(0)]
            for label, text in [(0, "Non-churners"), (1, "Churners")]:
                axes[row, column].hist(
                    frame.loc[frame.churn.eq(label), "score"],
                    bins=np.linspace(0, 1, 21),
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    label=text,
                )
            axes[row, column].set_title(f"{name}; repeat {repeat}")
            axes[row, column].set_xlabel("Score (mixed outer models)")
            axes[row, column].set_ylabel("Within-class density")
    axes[0, 0].legend()
    figure.suptitle("Baseline score overlap — descriptive development predictions")
    figure.tight_layout()
    figure.savefig(output / "score_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, subgroup in zip(axes, ["overall", "recent_purchase"]):
        subset = curves[curves.subgroup.eq(subgroup)]
        for (repeat, fold), frame in subset[subset.scope.eq("outer_fold")].groupby(
            ["repeat", "outer_fold"]
        ):
            ax.step(
                frame.recall, frame.precision, where="post", color="gray", alpha=0.20
            )
        for repeat, frame in subset[
            subset.scope.eq("repeat_pooled_mixed_models")
        ].groupby("repeat"):
            ax.step(
                frame.recall,
                frame.precision,
                where="post",
                label=f"Repeat {repeat}, mixed models",
            )
        ax.axvline(0.90, color="black", linestyle=":", alpha=0.5)
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=subgroup.replace("_", " ").title(),
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    figure.suptitle("Gray: individual outer folds; colored: descriptive pooled curves")
    figure.tight_layout()
    figure.savefig(output / "precision_recall.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    overall = metrics[(metrics.scope == "repeat_pooled") & metrics.slice.eq("overall")]
    for policy, frame in overall.groupby("policy"):
        axes[0].plot(frame.repeat.astype(str), frame.recall, "o-", label=policy)
        axes[1].plot(frame.repeat.astype(str), frame.fp, "o-", label=policy)
    axes[0].set(ylabel="Recall", xlabel="Repetition", ylim=(0, 1))
    axes[1].set(ylabel="False alerts", xlabel="Repetition")
    axes[0].legend()
    figure.suptitle("Same baseline scores, different saved operating policies")
    figure.tight_layout()
    figure.savefig(output / "operating_policies.png", dpi=160)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Small dependency-free report tables."""
    frame = frame[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                "undefined"
                if pd.isna(value)
                else f"{value:.4f}"
                if isinstance(value, float)
                else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    burden: pd.DataFrame,
    stability: pd.DataFrame,
    verification: dict,
    output: Path,
) -> None:
    overall = metrics[(metrics.scope == "repeat_pooled") & metrics.slice.eq("overall")]
    activity = metrics[
        (metrics.scope == "repeat_pooled") & metrics.slice.eq("activity")
    ]
    pattern = (
        stability.groupby(["policy", "error_pattern"])
        .agg(account_snapshots=("StoreID", "size"), resellers=("StoreID", "nunique"))
        .reset_index()
    )
    uncertain = (
        stability.groupby("policy")[["rank_unstable", "decision_unstable"]]
        .sum()
        .reset_index()
    )
    cutoff = (
        cases.groupby(["policy", "repeat", "outer_fold"])
        .cutoff.first()
        .groupby("policy")
        .agg(["min", "median", "max"])
        .reset_index()
    )
    slice_summary = (
        metrics[
            (metrics.scope == "repeat_pooled")
            & metrics.slice.isin(["annual_orders", "history_span", "snapshot_slice"])
        ]
        .groupby(["policy", "slice", "group"])[
            ["recall", "precision", "false_positive_rate", "fp", "fn"]
        ]
        .mean()
        .reset_index()
    )
    ranking = (
        metrics[
            (metrics.scope == "outer_fold")
            & metrics.slice.isin(["overall", "activity"])
        ]
        .groupby(["policy", "slice", "group"])[["average_precision", "roc_auc"]]
        .mean()
        .reset_index()
    )
    rank_primary = ranking[ranking.policy.eq("inner_selected")]
    persistent = (
        stability[stability.error_pattern.eq("consistent_error")]
        .groupby(["policy", "activity", "churn"])
        .size()
        .rename("account_snapshots")
        .reset_index()
    )
    burdens = burden[burden.scope.eq("repeat_pooled_mixed_models")]
    activity_means = activity.groupby(["policy", "group"])[
        ["recall", "precision", "false_positive_rate", "fp"]
    ].mean()
    active = activity_means.loc[("inner_selected", "recent_purchase")]
    inactive = activity_means.loc[("inner_selected", "no_recent_purchase")]
    consistent = stability[
        (stability.policy == "inner_selected")
        & stability.error_pattern.eq("consistent_error")
    ]
    consistent_fp = int(consistent.churn.eq(0).sum())
    consistent_fn = int(consistent.churn.eq(1).sum())
    active_consistent_fp = int(
        (consistent.churn.eq(0) & consistent.activity.eq("recent_purchase")).sum()
    )
    rank_unstable = int(
        stability[(stability.policy == "inner_selected")].rank_unstable.sum()
    )
    missed = cases[
        (cases.policy == "inner_selected") & cases.result_type.eq("missed_churn")
    ]
    mature_misses = int(missed.annual_orders.eq("3+").sum())
    total_misses = len(missed)
    policy_means = overall.groupby("policy")[["tp", "fp", "recall", "precision"]].mean()
    added_tp = (
        policy_means.loc["inner_selected", "tp"] - policy_means.loc["fixed_006", "tp"]
    )
    added_fp = (
        policy_means.loc["inner_selected", "fp"] - policy_means.loc["fixed_006", "fp"]
    )
    rate_summary = slice_summary[
        (slice_summary.policy == "inner_selected")
        & slice_summary.slice.eq("snapshot_slice")
    ]
    weakest_fpr = rate_summary.sort_values("false_positive_rate", ascending=False).iloc[
        0
    ]
    observations = f"""
- **The recent-purchase group remains difficult.** Under inner-selected cutoffs,
  mean repeat recall is {active.recall:.2%}, precision {active.precision:.2%}, and
  false-positive rate {active.false_positive_rate:.2%}. Accounts without a recent
  purchase have recall {inactive.recall:.2%}, but false-positive rate
  {inactive.false_positive_rate:.2%}: all continuing buyers in that group are alerted.
- **Many errors persist across fits.** There are {consistent_fp} account/snapshot
  cases falsely alerted in all three repetitions ({active_consistent_fp} recently
  purchasing), and {consistent_fn} churn cases missed in all three. These are
  distinct cases, not counts of repeated predictions; see account_stability.csv.
- **Sparse history alone does not explain the misses.** {mature_misses} of
  {total_misses} missed-churn events across repetitions have three or more annual
  orders. This is a dependent event count, not independent churners. Group sizes
  and rates are needed to interpret it; more orders do not guarantee separability.
- **Operating policy has a substantial cost.** Compared with 0.06 on identical
  scores, the saved high-recall policy catches {added_tp:.2f} more churn cases per
  repetition on average but adds {added_fp:.2f} false alerts. No ranking changed.
- **Stability and snapshot variation matter.** {rank_unstable} of 1,375 cases move
  at least 20 rank-percentile points across fits. The highest snapshot-level
  mean false-positive rate under the high-recall policy is
  {weakest_fpr.false_positive_rate:.2%} in {weakest_fpr.group}. These grouped
  snapshot slices are descriptive, not future-period validation.
"""
    text = f"""# Baseline development error diagnosis

This report describes the original ten-feature model's saved held-out predictions.
No models were fitted, new cutoffs selected, candidates compared, or October
outcomes accessed. All four approved development snapshots are included. The
population is 1,375 account/snapshot cases from 422 resellers, with 212 churn cases.
Each case has three dependent held-out predictions; this is not 4,125 independent cases.

## Main observations

{observations}

## Verified operating results

The fixed 0.06 rule is a retrospective historical reference. The inner-selected
rule uses the saved training-only policy maximizing precision with recall >90%.
It is not a newly selected cutoff or the original 30%-precision selection policy.

{markdown_table(overall, ["policy", "repeat", "tp", "fp", "fn", "recall", "precision", "false_positive_rate"])}

## Recent purchasing activity

Recent purchase means at least one accepted order during the preceding three
months. No recent purchase still means eligible under the six-month rule.

{markdown_table(activity, ["policy", "repeat", "group", "rows", "churners", "tp", "fp", "fn", "recall", "precision", "false_positive_rate"])}

## Other predefined profiles

The following values are arithmetic means of repeat-level metrics, not an
independent validation set. Complete per-repeat/per-fold counts and discount
availability slices are in [subgroup_metrics.csv](subgroup_metrics.csv).
Observed history span is last minus first purchase inside the twelve-month
window, not true relationship tenure. A one-order history has span zero.

{markdown_table(slice_summary, ["policy", "slice", "group", "recall", "precision", "false_positive_rate", "fp", "fn"])}

## Ranking around high recall

First, mean within-fold ranking metrics (folds equally weighted; not independent):

{markdown_table(rank_primary, ["slice", "group", "average_precision", "roc_auc"])}

Below are descriptive mixed-model pooled curves at predeclared recall levels.
Held-out labels locate these points; they MUST NOT become deployment cutoffs.
Tied scores enter together, so achieved recall can exceed the target. Each
fold's separate curve and burden are retained in the CSVs. Policies share
identical scores and therefore identical ranking.

{markdown_table(burdens, ["repeat", "subgroup", "target_recall", "achieved_recall", "tp", "fp", "precision"])}

## Persistent versus unstable errors

Patterns refer to the same reseller/snapshot across three repetitions. Consistent
means wrong in 3/3; recurring means 2/3; occasional means 1/3. Counts across
snapshots of one reseller are dependent. Rank percentiles use midranks within
the complete held-out fold; ties are not separated by account ID.

{markdown_table(pattern, ["policy", "error_pattern", "account_snapshots", "resellers"])}

Consistent errors by activity and actual outcome (churn=0 means false alerts):

{markdown_table(persistent, ["policy", "activity", "churn", "account_snapshots"])}

Rank instability means a range of at least 20 percentile points across fits.
Decision instability means alerts in some but not all repetitions. These are
descriptive flags, not tests of statistical significance. Each repetition also
changes the held-out comparison pool; percentile variation therefore combines
model/refit changes with fold composition and cannot isolate model instability.

{markdown_table(uncertain, ["policy", "rank_unstable", "decision_unstable"])}

Saved outer-fold cutoffs:

{markdown_table(cutoff, ["policy", "min", "median", "max"])}

Changing a cutoff explains decision differences on identical scores. This audit
cannot isolate why scores differ across refits or prove that a new cutoff will
transfer to a full-data model. A further controlled stability experiment would
be required for that question.

## Inspection files and plots

- [Account-level diagnostics](account_diagnostics.csv): features, profiles, ranks,
  both decisions, and distance to the alert boundary. Near-boundary means within
  five percentile points. High-ranked false alerts are flagged in the top 25%;
  low-ranked missed churners in the bottom 25%. These are review categories.
- [Account stability](account_stability.csv): recurrence and policy agreement.
- [Rank-band metrics](rank_band_metrics.csv) and [ranking curves](ranking_curves.csv).
- [Descriptive high-recall burden](high_recall_burden.csv).
- [Score overlap](score_distributions.png), [precision-recall curves](precision_recall.png),
  and [operating-policy comparison](operating_policies.png).

## Interpretation and next questions

Observations must be supported by the tables above. Persistent false alerts and
misses identify useful cases for step two: compare their pre-snapshot histories
with similar correctly classified accounts. The annual-order/history-span and
missingness slices describe associations; they do not establish causes, feature
usefulness, or which data additions should be selected. No feature or replacement
model is selected here.

Questions for step two are whether omitted purchasing changes distinguish the
recurring errors, whether sparse histories explain those cases, and whether
decision changes primarily follow cutoffs or unstable ordering. Review existing
feature experiments before defining new candidates. Future comparisons must
match or exceed baseline recall on the same validation splits while reducing
false alerts; merely exceeding the older 90% recall rule is insufficient.

## Verification and limitations

{verification["saved_metric_rows_reproduced"]} saved baseline metric rows were
independently reproduced; all 15 outer splits and saved inner cutoff provenance
were audited. Input and output identities are in manifest.json. This checks
saved predictions and metadata, not estimator refits or an independent fresh
source-completeness audit. Small samples (rows <30, churners <10, or non-churners
<10) are flagged; empty or undefined rates remain missing. No confidence intervals
are claimed. Grouped snapshots describe unseen identities in mixed historical
periods, not future-period generalization. Development has been inspected before;
these findings are exploratory and cannot establish an independent improvement.
"""
    (output / "report.md").write_text(text, encoding="utf-8")

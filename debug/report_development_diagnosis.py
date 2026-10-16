"""Render a development-only error diagnosis from completed saved predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_saved_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["october_test_outcomes_used"] is False
    cases = read_saved_csv(directory / "current_development_cases.csv")
    metrics = read_saved_csv(directory / "development_pooled_metrics.csv")
    folds = read_saved_csv(directory / "development_metrics.csv")
    missed = cases[cases.result.eq("missed_churn")]
    # The narrative below documents this specific completed experiment. Reject
    # a different run rather than silently attach stale numbers to new results.
    assert len(cases) == 1375 and cases.churn.sum() == 212
    assert len(missed) == 52 and missed.active.sum() == 50
    expected = {
        ("grouped", "current", "fixed_006"): (160, 324),
        ("grouped", "current", "inner_recall_over_90"): (195, 530),
        ("grouped", "behavioral", "inner_recall_over_90"): (197, 493),
        ("temporal", "current", "fixed_006"): (46, 155),
        ("temporal", "current", "inner_recall_over_90"): (55, 201),
        ("temporal", "behavioral", "inner_recall_over_90"): (59, 204),
    }
    for row in metrics[metrics.subgroup.eq("overall")].itertuples():
        assert (row.tp, row.fp) == expected[(row.method, row.candidate, row.policy)]
    summary = (
        cases.groupby(["active", "result"])
        .agg(
            rows=("StoreID", "size"),
            median_recent_revenue=("revenue_3m", "median"),
            median_annual_revenue=("revenue_12m", "median"),
            median_recency=("recency_days", "median"),
            median_annual_orders=("n_orders_12m", "median"),
            median_revenue_ratio=("revenue_ratio_3m", "median"),
        )
        .reset_index()
    )
    summary.to_csv(directory / "error_summary.csv", index=False)
    # Per-fold decisions already freeze the selected threshold before outer scoring.
    predictions = read_saved_csv(directory / "development_predictions.csv")
    current = predictions[
        predictions.method.eq("grouped")
        & predictions.candidate.eq("current")
        & predictions.policy.eq("inner_recall_over_90")
    ]
    behavioral = predictions[
        predictions.method.eq("grouped") & predictions.candidate.eq("behavioral")
    ]
    keys = ["StoreID", "snapshot", "churn"]
    paired = current[keys + ["score", "cutoff"]].merge(
        behavioral[keys + ["score", "cutoff"]],
        on=keys,
        suffixes=("_current", "_behavioral"),
        validate="one_to_one",
    )
    before = paired.score_current.ge(paired.cutoff_current)
    after = paired.score_behavioral.ge(paired.cutoff_behavioral)
    positive = paired.churn.eq(1)
    changes = dict(
        newly_caught=int((positive & ~before & after).sum()),
        newly_missed=int((positive & before & ~after).sum()),
        removed_false_alerts=int((~positive & before & ~after).sum()),
        added_false_alerts=int((~positive & ~before & after).sum()),
    )
    (directory / "paired_changes.json").write_text(json.dumps(changes, indent=2) + "\n")
    overall = metrics[metrics.subgroup.eq("overall")]
    result_table = [
        "| Validation | Model / cutoff policy | Recall | Precision | Caught | False alerts |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overall.itertuples():
        label = (
            "Current, original 0.06"
            if row.policy == "fixed_006"
            else (
                "Current, training-selected cutoff"
                if row.candidate == "current"
                else "Behavioral additions, training-selected cutoff"
            )
        )
        result_table.append(
            f"| {row.method} | {label} | {row.recall:.2%} | {row.precision:.2%} | {row.tp} | {row.fp} |"
        )
    target_folds = folds[
        folds.subgroup.eq("overall") & folds.policy.eq("inner_recall_over_90")
    ]
    fold_table = [
        "| Validation | Candidate | Folds above 90% recall |",
        "| --- | --- | ---: |",
    ]
    for (method, candidate), group in target_folds.groupby(["method", "candidate"]):
        fold_table.append(
            f"| {method} | {candidate} | {int(group.recall.gt(0.9).sum())}/{len(group)} |"
        )
    doc = f"""# Development diagnosis: precision with recall above 90%

The current model struggles to separate recently active churners from retained
accounts, particularly low-spending retained accounts. A six-feature behavioral
addition improves the grouped development tradeoff slightly, but fails to sustain
90% recall in chronological validation. No improved model is selected.

## Scope and selection rule

This report uses only the approved July/October 2012 and January/April 2013
development snapshots: 1,375 account/snapshot rows, 422 distinct resellers, and
212 churn cases. No October 2013 scoring rows or test labels are used by this run.
The user requires precision improvement with **recall strictly above 90%**.
Comparison of development metrics with the old test precision is not evidence
of an improvement; compare candidates on the same validation rows and policy.

The first exploratory reproduction earlier in this conversation inspected the
already inspected October cohort. It is set aside and is not a model-selection
input here. This does not restore its independence. These are exploratory
development results, and an eventual final model needs a new untouched cohort.

## Which churners are hard?

At the inherited 0.06 cutoff, held-out reseller predictions catch 160 of 212
churn cases, miss 52, and produce 324 false alerts (75.47% recall, 33.06%
precision). Of the misses, **50/52 are recent purchasers**. These are cases,
not 52 distinct people: the misses span {missed.StoreID.nunique()} resellers.

Among all 52 misses, {int(missed.recency_days.le(31).sum())} purchased in the
preceding 31 days. Quarter-over-quarter spending is flat or rising for
{int(missed.revenue_ratio_3m.ge(1).sum())}, falls at least 25% for
{int(missed.revenue_ratio_3m.le(0.75).sum())}, has a smaller decline for
{int(missed.revenue_ratio_3m.between(0.75, 1, inclusive="neither").sum())}, and
has no positive previous-quarter denominator for
{int(missed.revenue_ratio_3m.isna().sum())}. Absence of a ratio is not flat spending.

Recently active missed churners have median recency 32 days, four orders in the
past year, recent spending 2,275.86, annual spending 13,664.07, and an observed
median recent/previous-quarter revenue ratio of 1.108. They are not uniformly
dormant, declining, or lacking history. Purchase history alone may not reveal
an imminent stop; that is a hypothesis about information limits, not proof.

The full account/snapshot list is `development_missed_churners.csv`.

## Why so many false alerts?

The active subgroup contains 1,221 rows and 121 churn cases. The current rule
catches 71, misses 50, and raises 263 false alerts: 58.68% recall and 21.26%
precision. Inactive rows have 89 caught, two missed, and 61 false alerts.

For active rows, median recent spending is 744.27 among false alerts, 1,308.94
among caught churners, 2,275.86 among missed churners, and 19,662.66 among
correctly retained accounts. All four groups have median recency 32 days.
Low absolute spending aligns with many alerts but does not distinguish all
churners from small continuing accounts. These profiles are associations;
they do not establish a causal effect or isolate the estimator's dependence.

The active churn rate shifts from 13.98% in July to 4.73% in October, 4.18%
in January, and 17.20% in April. The inactive rate also changes. A mixed-period
grouped score can therefore hide substantial difficulty predicting a later
period. Grouping answers an unseen-reseller question; chronological validation
answers the later-period question, with smaller available training samples here.

## What was tested?

The original ten features, balanced training-only weights, fixed histogram
boosting parameters, and five seeds remain the control. The candidate adds:

- recent/previous-quarter revenue ratio;
- recent/previous-quarter order-count ratio;
- days overdue relative to the mean observed purchase gap;
- annual order count;
- purchase-gap coefficient of variation;
- latest-order revenue relative to mean annual order size.

Missing discount values and undefined behavioral ratios remain missing.
All features use only pre-snapshot transactions. No operational model is changed.

Five outer StratifiedGroupKFold folds keep each StoreID wholly outside its
training fold. Three inner grouped folds in each outer-training set select a
cutoff maximizing precision while requiring **inner pooled recall >90%**.
Ties favor recall, then the higher cutoff. Each frozen cutoff is applied to
the outer validation population. Neither outer labels nor test labels select it.
Both the current and behavioral model get this same cutoff policy, isolating
the feature comparison from a cutoff-only comparison.

Temporal checks score January and April separately, training only on snapshots
whose full six-month labels have matured by the validation date. January uses
July; April uses July and October. Inner folds mix these available training
periods, but outer future labels never enter fitting or cutoff selection.

{chr(10).join(result_table)}

Grouped pooled scores combine disjoint outer validation predictions with
fold-specific cutoffs; there is no single selected deployment cutoff.
Temporal pooled scores combine two periods and must be read alongside each
period's results in `development_metrics.csv`.

At the high-recall policy, behavioral features newly catch {changes["newly_caught"]}
churn cases and newly miss {changes["newly_missed"]}; they remove
{changes["removed_false_alerts"]} false alerts and add {changes["added_false_alerts"]}.
The net change is two more caught and 37 fewer false alerts. This yields
**+0.94 percentage points recall and +1.65 points precision**, relative to the
current model under the same nested selection policy.

{chr(10).join(fold_table)}

The behavioral model fails 90% recall in one of five grouped outer folds and
both temporal periods. Its temporal recall is 88.00% in January and 57.81%
in April. Its pooled temporal precision is 22.43%, and its pooled average
precision is 0.2168 versus 0.2470 for the current model. Thus the modest grouped
gain is not a reliable solution to the requested operating target.

The five boosting seeds give identical predictions in all observed fits
(maximum within-ensemble score range zero). Averaging these seeds contributes
no observed prediction diversity here; five seeds are not five independent
confirmations. Split and time sensitivity matter more.

## Recommended development work

1. Freeze the objective as maximum precision subject to recall >90%, evaluated
   on outer validation predictions. Report every period, active/inactive subgroup,
   alert count, and paired gained/lost churner. Treat feasibility failures as
   failures rather than averaging them away. Set a useful precision improvement
   margin before a larger search; a tiny numerical gain is not automatically
   operationally useful.
2. Obtain more historical scoring dates with fully matured development labels.
   Four snapshots and two chronological validation dates cannot support strong
   claims of stable future recall. More rows from the same few dates do not
   substitute for more time regimes. Never add labels that were unavailable
   at a simulated prediction date.
3. Keep behavioral normalization as a candidate, not a winner. Compare compact
   ablations and a regularized baseline under repeated nested grouped validation
   and predeclared chronological folds. The current addition improves one
   validation question but worsens pooled temporal ranking quality.
4. Test an active/inactive model interaction or separate risk heads, with a
   shared or partially pooled estimator if subgroup samples are too small.
   Choose all thresholds using inner development predictions under the global
   recall constraint. Dropping active alerts indiscriminately can miss the
   very churners that make the target difficult.
5. Investigate forward-available relationship signals: quote/inquiry activity,
   sales contact changes, credit/payment deterioration, service issues, and
   lost/rejected orders. These may help distinguish active accounts about to
   stop when completed purchases still look normal. They are untested hypotheses
   and need verified recording timestamps. Do not import eventual churn reasons
   or post-snapshot status as historical predictors.
6. Freeze the entire chosen procedure and cutoff before one final untouched
   evaluation. Calibrate only from development if interpretable probabilities
   are needed; calibration alone does not establish a better precision/recall
   frontier. Neither a higher cutoff nor a different revenue total is a proven
   solution to the separation problem.

## Verification and limits

Source headers and lines reconcile against accepted clean orders; declared
coverage and export hashes are checked. All seven original base features and
all 1,375 development labels match the tracked research reference within stated
tolerances. A saved original expanded ten-feature artifact is unavailable;
exact historical model/prediction identity is not claimed.

`development_protocol.json` records the candidate specification before fitting.
`manifest.json` records source/code/runtime identities and output checksums;
`model_identities.json` records the fitted-estimator hashes, seeds, and training
content hashes. Generated data and models remain ignored.

The runner excludes October test scoring and outcomes. Earlier diagnostic
artifacts are retained separately for audit and are not inputs to this run.
One grouped partition, small temporal samples, correlated reseller snapshots,
earlier development inspection, and source-availability assumptions limit the
strength of the evidence. No independent final test or successful replacement
model is claimed.

Six focused synthetic tests cover temporal feature isolation, native missing
discounts, half-open development labels, protected-snapshot rejection, strict
recall cutoff selection, equality at the alert boundary, and saved-score float
round-trip behavior. The repository-wide
pytest invocation is blocked before collection by pre-existing duplicate TOML
sections; the package also has a pre-existing committed merge conflict in
`reseller_churn/__init__.py`. Those files were left unchanged.
"""
    (directory / "diagnosis.md").write_text(doc, encoding="utf-8")
    inputs = [
        "manifest.json",
        "current_development_cases.csv",
        "development_pooled_metrics.csv",
        "development_metrics.csv",
        "development_predictions.csv",
    ]
    outputs = ["diagnosis.md", "error_summary.csv", "paired_changes.json"]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (directory / "report_manifest.json").write_text(
        json.dumps(
            {
                "renderer_sha256": digest(Path(__file__)),
                "inputs": {name: digest(directory / name) for name in inputs},
                "outputs": {name: digest(directory / name) for name in outputs},
            },
            indent=2,
        )
        + "\n"
    )
    print(directory / "diagnosis.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)

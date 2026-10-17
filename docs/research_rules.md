# Experiment lessons and rules moving forward

Updated 2026-10-17. This document owns the research and reporting rules agreed
after the grouped studies, ablations, October evaluation and failure diagnosis.
It documents future requirements; it does not claim that every requirement
is already enforced by the experiment scripts. Historical protocols and their
measured results remain intact. No model is promoted by this document.

## What the experiments established

The current user requirement is **reduce false alerts while matching or
exceeding baseline recall on the same development validation splits**. Future
comparisons must predeclare the common cutoff-selection policy and the exact
paired acceptance calculation before fitting. The historical studies below
used pooled grouped recall strictly above 90% in every repetition; their rules
and recorded results are unchanged. Neither requirement guarantees performance
for every subgroup or new population. Grouped validation remains the agreed
selection design.

| Evidence | Recall | Precision | What it establishes |
| --- | ---: | ---: | --- |
| Development baseline, inner-selected high-recall cutoffs | 91.51% | 27.16% | Reference under the development selection policy |
| Development adaptive selection procedure | 91.51% | 29.74% | Improvement for the procedure across saved grouped repeats |
| Development fixed cadence/regularization configuration | 94.65% | 29.31% | Performance of that configuration across outer refits and inner-selected cutoffs |
| October original model, historical 0.06 rule | 93.48% | 28.48% | Historical operating-rule result on the inspected cohort |
| October frozen cadence candidate | 97.83% | 20.09% | Failure to improve precision over that historical rule |

The two development candidate results use the same development population;
they are not two independent validation sets. The adaptive procedure may
choose a different configuration in each outer fold. The fixed-configuration
row summarizes one configuration, but still uses different fitted outer models
and training-selected cutoffs. Neither is an independent validation of the
single full-development fitted bundle. Its configuration was selected using
development evidence, so that row must not be presented as an independent
confirmation after selection.

The development gain did not establish a general model-quality improvement.
At 29.31% precision, about seven in ten alerts were already false. The candidate
caught two extra October churners at the cost of 71 net extra false alerts.
It remains a research artifact, not an accepted precision-improving replacement.

## What the diagnosis supports and leaves unresolved

- Candidate active recall and false-positive rate are nearly unchanged between
  grouped validation (91.18% / 38.42%) and October (90.91% / 38.76%). Active churn
  prevalence falls from 9.91% to 2.46%. Applying development active/inactive
  operating rates to October counts gives 20.17% overall precision, close to
  the observed 20.09%. This is descriptive arithmetic, not causal proof or an
  independent forecast.
- Both October models face the same population. The original rule has fewer
  active false alerts (98 versus 169). Prevalence alone cannot explain the
  candidate's disadvantage to the original rule.
- The development baseline uses inner-selected high-recall cutoffs; the
  historical October baseline uses 0.06. Those result differences do not
  isolate feature and parameter effects under a common cutoff policy.
- Aggregate gains conceal weaker groups. Active precision decreases in two
  of four development snapshots. Overall October AP improves while active AP
  worsens. Strong overall recall includes many already inactive churners.
- Existing development ablations support gap variation and overdue ratio as
  precision contributors. Smaller trees mainly help recall. Annual count and
  L2 effects remain uncertain after adjustment. These are context-dependent
  development findings, not proof of which change caused October's failure.
- No saved-artifact population, score or changed-cutoff error was found.
  The exact original fitted artifact is unavailable; the baseline is a local
  reproduction. Outcomes reuse an earlier accepted-orders reconstruction;
  source completeness was not freshly audited. Refit/cutoff-transfer stability,
  individual-change causality and future-population performance remain unresolved.

See [the complete diagnosis](../comparisons/frozen_failure_diagnosis.md),
[metrics](evaluation_metrics.md), and [ablations](../comparisons/grouped_ablation.md).

## Current development decisions — 2026-10-17

The goal remains improved model performance developed without the final test.
Repeated validation use is normal development work; it is not itself an error.
We will keep diagnosing development errors and freeze each bounded comparison,
reporting uncertainty and adverse periods rather than treating a favorable mean
as sufficient. No guaranteed future performance follows from a development score.

Temporal validation has already been tried. Grouped selection led the controlled
April2013 comparison on points,with an inconclusive formal winner. Mixed-period
validation limits forecasting claims,but is not an identified cause of failure;
automatically switching to temporal selection is not a supported remedy. Keep
both evidence types distinct and their different populations/training sizes clear.

The [development preparation protocol](../comparisons/development_research_protocol.md)
owns the next-fit freeze requirements and implemented APIs. New comparisons must
report and predeclare limits for false-alert harm AND recall in each period and
activity group,per repeat. Proposed research defaults are zero additional FP and
at most five percentage points of recall loss in those groups; overall recall
must still match baseline. They are prospective research defaults,not approved
business margins or changes to historical decisions. Every fold and period/activity
intersection is also reported,including sparse/undefined metrics. Freeze limits
before fitting and pass them explicitly; do not average a failed period away.

October rows,labels,predictions and errors stay outside further development choices.
No new training or final evaluation is authorized by this preparation document.

## Rules for the next development experiment

1. **Freeze the comparison before execution.** Record candidates, feature
   formulas, historical information boundaries, model/ensemble parameters,
   grouped splits, cutoff-selection policy, tie rules, primary objective,
   uncertainty method and acceptance criteria. Preserve input/code/model
   identities and report every planned comparison, including failures.
2. **Use matched comparisons.** Give methods the same rows, reseller groups,
   folds and cutoff-selection objective. Report the original fixed 0.06 rule
   separately. Any different selection policy is an explicitly different method;
   do not attribute its entire metric difference to features or parameters.
3. **Keep selection inside training.** Use nested reseller-grouped validation
   for features, parameters and cutoffs; derive class weights and preprocessing
   from each training fold only. Preserve native missing discount/cadence values
   and historical feature boundaries. All snapshots of one reseller stay
   together. Repeatedly choosing changes after inspecting outer results makes
   those results exploratory, even when each individual run is nested.
4. **Report performance beyond the average.** Include each repeat and fold,
   pooled and subgroup TP/FP/FN/TN, precision, recall, false-positive rate,
   alert counts, churn rates and ranking metrics. Include active/inactive groups
   and consistently defined behavioral and development-snapshot slices.
   Snapshot slices remain descriptive grouped results, not temporal validation.
   Report subgroup sample sizes and deterioration; do not hide an aggregate
   failure or select only favorable slices.
5. **Check robustness before calling a candidate successful.** For new comparisons,
   apply the user's paired baseline-recall requirement above; exceeding 90% alone
   is insufficient. Examine uncertainty and variation
   in precision and false alarms, plus cutoff/refit stability within development.
   Freeze any additional practical-gain margin, subgroup constraint or worst-fold
   requirement before a new experiment. None is silently imposed here, and no
   development rule guarantees recall above 90% on unseen accounts.
6. **Stress-test precision without changing the datasets.** Predefine a range
   of hypothetical churn rates using development or independently available
   business information. Report precision implied by development recall/FPR at
   those rates, with the explicit assumption that those operating rates hold.
   Do not choose the range or tune the model to match October outcomes. Such
   calculations expose prevalence sensitivity; they do not simulate all changes
   in feature/outcome relationships or replace independent evaluation.
7. **Do not force evaluation balance.** Keep validation/test populations at
   their natural distributions. Initial predefined stratification with reseller
   isolation is permissible; it does not guarantee matching future prevalence.
   Any resampling belongs within training folds only. Do not remove, duplicate
   or move evaluation accounts to manufacture matching churn rates.
8. **Measure feature usefulness on development data.** Use matched ablations
   and uncertainty, with effects interpreted in the complete feature context.
   Uncertain contribution does not establish uselessness or equivalence after
   removal. Do not select a feature set from October error profiles.

## Test access and honest claims

Evaluating a frozen method once is appropriate; inspecting the result does
not retroactively contaminate that frozen evaluation. Choosing subsequent
changes from test errors makes that cohort unsuitable for independently
evaluating those changes, even if its rows never enter model fitting.

October 2013 was inspected before the latest follow-up and is permanently
marked as inspected. Its evaluation and diagnosis are descriptive reuse.
Keep its outcomes outside training and selection. Do not repeatedly test
candidate alternatives on it, optimize its cutoff, balance it, or describe a
later result there as a fresh independent test. October-inspired hypotheses
must be identified as such, not disguised as independently chosen improvements.

Before any new independent final evaluation, freeze the complete methodology
and obtain explicit user approval for an appropriate untouched cohort. Predict
without accessing outcomes; save model/input/code identities and predictions
before a separate evaluation joins explicit supplied outcomes. No automatic
protected-label construction is supported. Do not tune after evaluation or
deploy a research candidate merely because development criteria passed.

Reports must label **development selection procedure**, **fixed-configuration
development summary**, **frozen-model evaluation**, **descriptive diagnosis**,
and **hypothesis** accurately. State the population, operating policy, prior
inspection, uncertainty and source limitations next to each result. Conditional
bootstrap intervals do not cover all retraining, selection or population-shift
uncertainty. Repeated folds and identical seed predictions are not independent
new evidence. Smoke tests establish execution correctness only.

Research stays on `pipeline_preparation`; final pipeline preparation stays on
`final-pipeline-preparation`. Preserve the operational fixed ten-feature
pipeline, tracked dataset references and unrelated edits. No automatic push,
merge, promotion or deployment is authorized by these findings.

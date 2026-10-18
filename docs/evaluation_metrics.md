# Evaluation metrics and latest results

Updated 2026-10-18. The objective is fewer false alerts while preserving matched
baseline recall under each experiment's frozen rules. Historical high-recall
studies maximize precision subject to recall >90%; later precision-target studies
maximize recall subject to a development precision floor. These are different
operating policies. Nested outer validation, threshold-selection OOF and October
results are identified separately below. Development studies exclude October;
subsequent user-authorized evaluations reuse that previously inspected cohort.
The original 2026-10-11 record and every historical acceptance decision remain.

The historical provisional pipeline's latest available evaluation files are dated
2026-10-07. Its results are recorded separately below because its models and
evaluation summaries differ. See [project_context.md](project_context.md) for
data roles and [feature_analysis.md](feature_analysis.md) for model inputs.

**Current conclusion:** no experimental replacement passes the complete frozen
development acceptance requirements. Personalized recency improves development
ranking, but its 40%-target October policy adds 17 false alerts for one additional
caught churner versus period/class alone. Results vary by operating target; a
favorable mean or inspected October result does not establish acceptance.
[Research rules](research_rules.md) own future comparison requirements.

## What the metrics mean

The positive class is **churn**: an eligible reseller places no order during the
following six months. An alert predicts churn and may prompt retention work.

| Outcome | Plain meaning |
| --- | --- |
| True positive (TP) | We alert a reseller who actually churns |
| False positive (FP) | We alert a reseller who continues ordering; a false alert |
| False negative (FN) | A reseller churns and we miss them |
| True negative (TN) | We do not alert a reseller who continues ordering |

| Metric | Formula / calculation | Simple interpretation |
| --- | --- | --- |
| Recall | `TP / (TP + FN)` | Of all actual churners, how many did we catch? Higher recall means fewer missed churners |
| Precision | `TP / (TP + FP)` | Of all alerted resellers, how many actually churned? Higher precision means less wasted outreach |
| False alerts | `FP` | Number of alerted resellers who did not churn; an operational workload count |
| Missed churners | `FN` | Number of actual churners we did not alert |
| F1 | `2 * TP / (2 * TP + FP + FN)` | A combined score that gives precision and recall equal weight; it does not encode the project's preference for recall |
| Accuracy | `(TP + TN) / all rows` | Fraction of all predictions that are correct; can look good when most resellers do not churn |
| False-positive rate | `FP / (FP + TN)` | Fraction of actual non-churners receiving an alert; different from the fraction of alerts that are false |
| PR-AUC / average precision | Average precision from the ranked churn scores | How well the model puts churners near the top across operating points; the project uses sklearn average precision, not trapezoidal integration of the precision-recall curve |
| ROC-AUC | Area under the true-positive-rate versus false-positive-rate curve | How well scores rank churners above non-churners; 0.5 is chance-level ranking and 1 is perfect ranking |
| Alert count | `TP + FP` | How many resellers the retention team would contact |
| Alert budget | `alert count / population` | Capacity expressed as a fraction of eligible resellers |
| Churn prevalence | `(TP + FN) / population` | How common churn is in the evaluated population; affects how precision and PR-AUC should be interpreted |

Precision, recall, F1, accuracy, and confusion counts depend on the rule used to
turn scores into alerts. Ranking metrics use scores across operating points.
The main threshold helper predicts churn when `score >= threshold`; the separate
controlled experiment instead alerts a fixed number of top-ranked resellers.
It has no fitted probability threshold. A score of 0.6 should not automatically
be interpreted as a calibrated 60% churn risk; calibration has not been verified.

The shared threshold helper reports zero for undefined precision/recall/F1 using
`zero_division=0`. The budget helper also returns zero recall if there are no
actual churners. Ranking metrics return `None` when average precision has no
positive examples or ROC-AUC lacks either class. Threshold reports include
explicit TP, FP, FN, TN, alert count, and alert rate. The historical `pr_auc`
key remains as a compatibility alias for the accurately named
`average_precision` value.

## Frozen candidate October follow-up — 2026-10-16

After development and ablations, the user explicitly authorized evaluating
the saved candidate. Its 13 features, seven leaves, L2=1, five-model ensemble,
and cutoff `0.21655335464238584` remain frozen. Evaluation fits no model and
optimizes no cutoff. Supplied outcomes are joined only after saving and
verifying an unlabeled prediction checkpoint.

| Rule | Recall | Precision | Caught | Missed | False alerts | Alerts | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 0.06 rule | 93.48% | 28.48% | 43 | 3 | 108 | 151 | 0.7013 |
| Frozen cadence candidate | 97.83% | 20.09% | 45 | 1 | 179 | 224 | 0.7892 |

Precision decreases 8.39 percentage points while recall increases 4.35 points.
Two additional caught churners cost 71 net additional false alerts. Both rules
catch all 35 inactive churners with ten false alerts. Active false alerts rise
from 98 to 169; active precision falls from 7.55% to 5.59% while recall rises
from 72.73% to 90.91%. Active AP falls from 0.1553 to 0.0765 while inactive AP
rises from 0.8870 to 0.9469. Aggregate ranking improvement does not establish
a better frozen operating point. The candidate **fails the requested precision
improvement**; no test-driven adjustment or replacement is made.

October outcomes were already inspected, so this is descriptive reuse, not a
new independent test. Baseline scores are a local reproduction matching
original counts; exact original artifact identity is unavailable. Outcomes
reuse an earlier accepted-orders reconstruction without fresh database
reconciliation. See [the evaluation record](../comparisons/frozen_october_evaluation.md)
for guards, provenance, commands, results, and independent verification.

### What the failure diagnosis adds

| Candidate, active resellers | Grouped validation | October |
| --- | ---: | ---: |
| Churn prevalence | 9.91% | 2.46% |
| Recall | 91.18% | 90.91% |
| False-positive rate | 38.42% | 38.76% |
| Precision | 20.70% | 5.59% |

Validation entries are means across repeats except prevalence, which describes
their shared population. Similar recall and false-positive rates can yield
very different precision when churn becomes rarer. At prevalence `p`, with
recall `r` and false-positive rate `f`, implied precision is
`r*p / (r*p + f*(1-p))`, assuming those operating rates remain applicable.
This does not account for every possible population or relationship change.

Applying development active/inactive rates to October subgroup counts gives
20.17% overall precision versus 20.09% observed. This is post-hoc explanatory
standardization, not a prospective estimate, model fix or instruction to
rebalance the test set. Both models face the same October population; the
candidate still produces more active false alerts than the original rule.

The development baseline uses inner-selected high-recall cutoffs, whereas the
October original uses 0.06. Do not attribute their cross-population differences
solely to added features or parameters. Future comparisons must use matched
policies and report any fixed-cutoff reference separately. Active precision
also declines in two development snapshots despite overall gains. See
[the diagnosis](../comparisons/frozen_failure_diagnosis.md) for the supporting
tables, unresolved causes, source limitations and verification scope.

## Grouped precision and ablation results — 2026-10-16

The development population has 1,375 account/snapshot rows, 422 resellers, and
212 churn cases across July/October 2012 and January/April 2013. Three repeat
seeds (42, 78, 88) each use five outer StratifiedGroupKFold folds; three inner
grouped folds use seed 78. Each reseller's snapshots stay together at both
levels. Training-only balanced weights and native missing handling are preserved.

For each configuration, inner OOF predictions select the cutoff maximizing
precision subject to pooled recall **strictly >90%**. Feature/parameter choices
also use only inner predictions. The selected configuration and cutoff are
then scored on the outer validation rows. This differs from the historical
recall-first, minimum-precision workflow below.

### Repeated grouped comparison

| Repeat seed | Baseline recall | Baseline precision | Adaptive recall | Adaptive precision | Fewer false alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 91.98% | 26.90% | 91.51% | 29.48% | 66 |
| 78 | 90.57% | 27.87% | 92.92% | 29.62% | 29 |
| 88 | 91.98% | 26.71% | 90.09% | 30.13% | 92 |
| Mean | 91.51% | 27.16% | 91.51% | 29.74% | 62.33 |

Baseline and adaptive procedures use the same inner cutoff objective. The
adaptive procedure selects among eight predefined feature/regularization
configurations inside training. Its mean precision gain is **2.59 percentage
points**, with a conditional paired reseller-cluster 95% interval of
**+1.67 to +3.52 points**. Each repetition's pooled recall exceeds 90%; only
11/15 individual outer folds do so, with minimum fold recall 71.43%.

These are arithmetic means of repeat-level pooled metrics. Every repetition
evaluates the same development population; 4,125 repeated predictions do not
constitute independent new cases. Each outer fold has a training-selected
cutoff, so the table does not evaluate one universal deployment cutoff.

Full-development inner selection subsequently freezes a 13-feature research
candidate: the original ten plus overdue ratio, annual order count, and gap
variation; seven leaves, L2=1, and cutoff `0.21655335464238584`. It is selected
by the evaluated procedure, but the table is not a separate validation of that
single full-data refit. See [the protocol and results](../comparisons/grouped_precision_recall.md).

The fixed configuration's separate grouped summary is 94.65% recall / 29.31%
precision. It uses the same development population, not a second independent
validation set. It evaluates that configuration across outer refits and selected
cutoffs; the adaptive row evaluates the selection process. Choosing or explaining
the fixed configuration after examining development results does not turn its
summary into independent confirmation of the final fitted model. Future reports
must name which object each number evaluates.

### Component ablation

Fourteen fixed configurations isolate five changes, using the same grouped
splits and inner cutoff objective. Seventeen contrasts cover single additions,
one-component removals, bundles, and interactions. Fifty thousand whole-reseller
bootstrap draws preserve each reseller's snapshots and repeat predictions;
Bonferroni percentile tails adjust precision intervals across the 17 contrasts.

| Component | Full-context precision inclusion effect | Adjusted interval | Full-context recall effect |
| --- | ---: | --- | ---: |
| Purchase-gap variation | +1.14 points | +0.36 to +1.98 | +0.94 points |
| Overdue ratio | +0.88 points | +0.17 to +1.63 | 0.00 points |
| Annual order count | +0.26 points | -0.28 to +0.83 | -0.63 points |
| Seven leaves rather than 31 | -0.08 points | -1.38 to +1.32 | +5.19 points |
| L2=1 rather than zero | +0.77 points | -0.003 to +1.52 | -0.94 points |

An inclusion effect is the complete candidate minus the candidate without that
component; positive means restoring it helps precision. Intervals are in
percentage points. Gap variation and overdue also have positive adjusted
precision intervals as standalone additions. Fewer leaves chiefly support
recall feasibility: removing that change lowers mean recall to 89.47% from
94.65%. Annual count and L2 remain uncertain; no tested precision interaction
has an adjusted interval excluding zero. Recall effects are descriptive
secondary evidence, without confirmatory multiplicity-adjusted claims.

The fixed complete candidate has mean recall **94.65%** and precision **29.31%**.
Gap variation alone gives **92.14% / 28.89%**, versus baseline **91.51% / 27.16%**,
and exceeds 90% recall in all repetitions. Its equivalence to the complete
candidate is unproven. Fixed-configuration results cannot be substituted for
the adaptive procedure's result or summed into shares of its gain.

On identical baseline scores, cutoff-only retuning changes recall from 77.36%
to 91.51% and precision from 32.96% to 27.16%, with unchanged AP. Gap variation
alone raises AP from 0.4843 to 0.5107. AP differences are descriptive ranking
evidence; decision effects include each configuration's inner cutoff adaptation.
See [the ablation record](../comparisons/grouped_ablation.md) for all contrasts,
error changes, interpretation, and reproduction commands.

Both studies reuse inspected development data. Cluster intervals condition on
saved fits/cutoffs and exclude retraining/reselection uncertainty; repetitions
are dependent. Bonferroni bootstrap intervals have approximate coverage and
do not undo earlier exploration. The ablation does not replace the research
candidate or access final testing. No new independent final result is claimed.

## Current development selection reports

This section describes the historical supported runner's precision-floor
sensitivity workflow. The latest grouped studies above use the opposite
constraint: maximize precision while recall exceeds 90%.

Two settings have different purposes:

- **Probability threshold:** the score cutoff used to flag churn. A threshold
  of 0.30 means an alert is raised for a score of 0.30 or higher.
- **Minimum precision (precision floor):** the required fraction of correct
  alerts in each validation fold. A minimum of 0.30 requires at least 3 in 10
  alerts to identify actual churners.

For each precision minimum, select the configuration and probability threshold
with the highest **mean recall across folds**, provided precision meets the
minimum in **every fold**. Break ties by worst-fold recall, mean precision, then
a stable configuration-and-threshold identifier. If no configuration qualifies,
report `no_feasible_candidate` without lowering the requirement.

The default precision minimums range from 0.10 to 0.90. They are experimental
settings, not approved business requirements. The workflow reuses validation
probabilities across all settings and produces one selection manifest per
minimum. Commands and the complete file list are in the
[pipeline guide](../pipeline/README.md).

Reports include per-fold confusion counts, precision, recall, F1, accuracy, and
alert rates, together with pooled results. **Mean fold recall** gives each
period equal weight; **pooled recall** divides total detected churners by total
actual churners across periods. The selection rule uses the former.

These validation scores inform selection, so they are not an independent
estimate of future performance. The implementation smoke run does not
establish a best model; its scope and checks are recorded in the
[refactoring record](refactoring_plan.md#verification).

The debug feature comparisons below use an agreed 30% precision floor.
Historical results are kept separately because their thresholds, selection
objectives, and evaluated populations differ.

## Final test evaluation — 2026-10-11

The user authorized final testing and froze ten-feature histogram gradient
boosting, baseline parameters, balanced training weights, five seeds averaged
as probabilities, and grouped-development cutoffs. The 30% floor is primary;
20%, 40%, and 50% floors are predefined secondary rules. The ensemble refits
on all 1,375 approved development rows. No parameters or thresholds change
after test outcome access.

The **2013-10-01** test contains **492 resellers and 46 churners**. Outcomes
refer to no purchase in **[2013-10-01, 2014-04-01)**. Final metrics are for one
cohort, not means across folds.

| Minimum development precision | Test recall | Actual test precision | F1 | Total churners | Churners caught | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% (secondary) | 97.83% | 20.64% | 0.341 | 46 | 45 | 173 |
| **30% (primary)** | **93.48%** | **28.48%** | **0.437** | **46** | **43** | **108** |
| 40% (secondary) | 71.74% | 56.90% | 0.635 | 46 | 33 | 25 |
| 50% (secondary) | 43.48% | 76.92% | 0.556 | 46 | 20 | 6 |

A churner caught is a correctly alerted account (TP). A false alarm is an
alert for an account that purchased during the outcome window (FP). Missed
churners equal total churners minus churners caught: 1, 3, 13, and 26,
respectively. A development precision floor fixes the selection rule; it does
not guarantee the same precision on a new cohort.

The primary cutoff remains 0.06. It catches **43 of 46** churners, misses three,
and generates **108 false alerts** among 151 alerts. It **does not meet 30%
precision on the test**, despite meeting that constraint in every grouped
development fold. The secondary
cutoffs remain 0.01, 0.55, and 0.88; their results are not used to select a new
test-derived winner.

Among recently active accounts, primary recall is 72.73% (8/11 churners) but
precision is 7.55% (eight true alerts plus 98 false alerts). Inactive accounts
contain 35 of the 46 churners, all caught. This population mix differs from
April development and limits interpreting high overall recall as a stable
future-period or subgroup result.

The model and unlabeled predictions were saved and hashed before labels were
revealed. All 896 reseller orders in the label window reconcile with the
CompanyX source. See [the frozen protocol](../results/final_evaluation/gb10_grouped_20261011/frozen_protocol.json),
[final metrics](../results/final_evaluation/gb10_grouped_20261011/final_test_metrics.csv),
and [full final-testing record](debugging.md#final-testing-evaluation--2026-10-11).
**This test is now inspected**; future changes using these results cannot claim
another untouched evaluation on the same outcomes. The model remains provisional.

<a id="latest-completed-experiment-seven-versus-ten-features"></a>

## Earlier seven-versus-ten feature experiment

The [experiment implementation](../debug/compare_seven_ten_features.py) tests
Logistic Regression, Random Forest, and histogram gradient boosting, using
fixed baseline parameters and balanced training weights. Seven features are
the Pruned baseline; ten add the three
[discount inputs](feature_analysis.md#experimental-ten-feature-set).

Each configuration selects one threshold maximizing mean fold recall with
**precision at least 30% in every overall validation fold**, over 0.00–1.00 in
0.01 steps. Tree scores average five fixed seeds; deterministic Logistic
Regression fits once per fold. The completed run has **154 fits** and 12 primary
configurations; all are feasible. Individual-seed diagnostics remain separate.

### Temporal validation: future-period development evidence

Two folds validate January and April 2013, with 683 account/snapshot cases and
89 churn cases. Training uses July 2012 for January and July/October 2012 for
April, with every training label resolved by prediction time.

| Model | Features | Threshold | Mean recall | Mean precision | Caught churn cases | False alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 7 | 0.72 | 35.03% | 35.03% | 23 | 44 |
| Logistic Regression | 10 | 0.74 | 35.03% | 36.25% | 23 | 41 |
| Random Forest | 7 | 0.59 | **36.59%** | 32.96% | 25 | 52 |
| Random Forest | 10 | 0.71 | 23.47% | 31.88% | 16 | 34 |
| Histogram Gradient Boosting | 7 | 0.93 | 10.78% | 44.44% | 6 | 6 |
| Histogram Gradient Boosting | 10 | 0.58 | 33.81% | 31.88% | 23 | 49 |

Discounts preserve Logistic Regression recall and save three false alerts.
They increase boosting recall in both periods, but reduce forest recall in
both. Seven-feature Random Forest has the highest mean temporal recall in
this experiment; its extra detected churn cases require more false alerts.

The April active-churner subgroup remains difficult: seven-feature forest catches
6/54 with five false alerts; ten-feature boosting catches 3/54 with seven false
alerts; Logistic Regression catches none. The 30% precision constraint is for
each overall fold, not every subgroup. Even the best of these subgroup results
still misses 48 April active churners.

### Grouped k-fold: held-out reseller evidence across mixed periods

Five folds use StratifiedGroupKFold with StoreID groups, shuffle=True, and
random_state=42. All snapshots of an account remain together. Validation covers
1,375 development rows, 422 reseller identities, and 212 churn cases.

| Model | Features | Threshold | Mean recall | Mean precision | Caught churn cases | False alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 7 | 0.49 | 54.31% | 38.26% | 115 | 189 |
| Logistic Regression | 10 | 0.49 | 56.18% | 38.74% | 119 | 193 |
| Random Forest | 7 | 0.26 | **75.55%** | 34.47% | 160 | 318 |
| Random Forest | 10 | 0.26 | 74.56% | 34.50% | 158 | 310 |
| Histogram Gradient Boosting | 7 | 0.13 | 70.86% | 37.39% | 150 | 257 |
| Histogram Gradient Boosting | 10 | 0.06 | 75.51% | 33.45% | 160 | 324 |

Ten-feature boosting's **75.51%** is mean fold recall. Its pooled recall is
`160 / 212 = 75.47%` and pooled precision is `160 / (160 + 324) = 33.06%`.
The different mean and pooled values reflect fold averaging. It generates
roughly two false alerts per caught churn case. Counts refer to account/snapshot
cases, not necessarily different resellers.

Adding discounts increases grouped mean recall for Logistic Regression and
boosting, with additional false alerts, but some individual folds lose recall.
The forest loses mean recall. Mean average precision decreases for all three
grouped models when adding discounts; higher constrained recall is not evidence
that ranking improves across all operating points.

### Interpretation, uncertainty, and evidence

Grouped and temporal scores are not comparable deployment estimates: they
evaluate different rows, training sizes, and chronology. Grouped folds may
train on later-period records than a held-out row and do not enforce historical
label availability. That earlier experiment treated temporal validation as
primary and retained both methods for reporting. The current precision-improvement
research uses grouped validation by user choice; these historical results do
not change that selection design.

Ten-feature boosting is worth further investigation, while seven-feature forest
remains the temporal comparison baseline. No architecture or feature set has
been promoted globally. Forest discount benefits vary by seed; boosting scores
are identical across the five seeds in this setup, which does not establish
independent replication. Threshold selection and repeated comparisons reuse
previously inspected development labels. These are exploratory results, not
significance claims or unseen-period guarantees. Protected final-test outcomes
were not accessed.

Primary evidence is the [selection summary](../results/debugging/seven_vs_ten_features/selection_summary.csv),
[per-fold/subgroup metrics](../results/debugging/seven_vs_ten_features/fold_and_subgroup_metrics.csv),
and [manifest](../results/debugging/seven_vs_ten_features/experiment_manifest.json).
The [debugging record](debugging.md#seven-versus-ten-features-across-models-and-validation-methods--2026-10-10)
owns detailed case changes, seed comparisons, earlier diagnoses, and verification.
Generated artifacts are ignored by Git and may be absent in a fresh checkout;
the [pipeline guide](../pipeline/README.md#debugging-and-feature-experiments)
lists prerequisites and reproduction commands.

## Historical controlled strategy comparison

The [experiment protocol](../comparisons/validation_strategy_experiment.md)
compares models selected by GroupKFold and temporal validation on **the same
340 April 2013 resellers**: 64 churners and 276 non-churners, or **18.82% churn**.

Both selectors search 36 registered configurations using April/October 2012
selection rows. Each selected model refits on the same 566-row pool, and its
five seed probabilities are averaged before common evaluation. Selection seeds
are 42, 78, 88, 1034, and 2026. April 2013 labels do not enter fitting or selection.

| Property | GroupKFold-selected model | Temporal-selected model |
| --- | --- | --- |
| Architecture | Random Forest | Random Forest |
| Feature set | Full, 16 sales-derived predictors | RFM, 3 predictors |
| Training imbalance treatment | None | Random oversampling |
| Model settings | Baseline: 300 trees, no maximum depth, minimum leaf size 1 | Same baseline settings |
| Seed probabilities | Average of five seeds | Average of five seeds |

These are experiment-specific selections, not replacements for the main tuned
seven-feature provisional models. The registered search omits `store_age`.

### Results at equal outreach capacity

The experiment alerts exactly `ceil(budget * 340)` resellers, ordering scores
descending and breaking score ties by ascending `StoreID`. Recall, precision,
and accuracy are percentages; F1 is on a 0–1 scale.

| Budget | Selection method | Alerts | Churners caught (TP) | False alerts (FP) | Missed churners (FN) | Recall | Precision | F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | GroupKFold | 34 | 17 | 17 | 47 | 26.56% | 50.00% | 0.3469 | 81.18% |
| 10% | Temporal | 34 | 9 | 25 | 55 | 14.06% | 26.47% | 0.1837 | 76.47% |
| 20% | GroupKFold | 68 | 24 | 44 | 40 | 37.50% | 35.29% | 0.3636 | 75.29% |
| 20% | Temporal | 68 | 21 | 47 | 43 | 32.81% | 30.88% | 0.3182 | 73.53% |
| 30% | GroupKFold | 102 | 34 | 68 | 30 | 53.13% | 33.33% | 0.4096 | 71.18% |
| 30% | Temporal | 102 | 32 | 70 | 32 | 50.00% | 31.37% | 0.3855 | 70.00% |

For example, GroupKFold at the 10% budget catches 17 of 64 actual churners
(`17 / 64 = 26.56%` recall). Of its 34 alerts, 17 are correct
(`17 / 34 = 50%` precision). The other 17 alerts are false, and 47 churners
are missed. This expresses both detection benefit and outreach cost.

At each fixed budget, more true positives also imply fewer false alerts because
the two models have the same alert count. Larger budgets caught more churners
in this run but required more outreach. These budgets are diagnostics rather
than approved business capacity limits.

Accuracy is included for completeness, not as the selection objective.
Predicting no churn for all 340 resellers would achieve `276 / 340 = 81.18%`
accuracy while catching no churners. This shows why accuracy alone is unsuitable
for a recall-prioritized churn project.

### Ranking results

| Selection method | Recorded PR-AUC (average precision) | ROC-AUC reconstructed from saved scores |
| --- | ---: | ---: |
| GroupKFold | 0.4183 | 0.7105 |
| Temporal | 0.2813 | 0.6503 |

PR-AUC is recorded in the original common-evaluation report. ROC-AUC was
calculated for this document from its saved development predictions, without
refitting. A no-skill precision reference for this population is the 18.82%
churn prevalence; PR-AUC should be interpreted relative to population prevalence
and the evaluation protocol, not as the percentage of churners caught.

### Uncertainty and interpretation

GroupKFold's mean recall across the three budgets is **39.06%**, compared with
**32.29%** for temporal selection: a **6.77 percentage-point** difference.
The saved 95% paired-bootstrap interval for that difference is **-5.88 to +16.67
percentage points**, using 5,000 reseller resamples and random seed 2026.
It includes zero, so the predeclared winner rule returns **inconclusive**.

GroupKFold's chosen model caught more churners at every reported budget, but
this sample does not establish a reliable advantage across future periods.
The interval is conditional on the selected configurations and observed period;
it does not include uncertainty from repeating model selection or from future
time periods. Five model seeds do not create five independent historical periods.

Additional limitations:

- April 2012 has only 306 observed days in the nominal 12-month feature window.
- April 2013 outcomes were inspected in earlier development work. This is a
  retrospective comparison, not an untouched held-out estimate.
- The selectors chose different feature sets and imbalance treatments. The
  result cannot isolate the causal contribution of validation design alone.
- No acceptable recall, precision, or outreach-cost target has been approved.
- Protected October 2013 churn outcomes were not evaluated.

### Evidence and reproducibility

- [Common metrics](../results/validation_strategy_experiment/common_evaluation_metrics.csv)
  supply alerts, TP, FP, recall, precision, F1, and PR-AUC.
- [Saved predictions](../results/validation_strategy_experiment/common_evaluation_predictions.csv)
  supply paired scores and development labels; ROC-AUC is derived from these.
- [Frozen manifest](../results/validation_strategy_experiment/frozen_selection_manifest.json)
  records selected models, feature inputs, budgets, and seeds.
- [Summary](../results/validation_strategy_experiment/experiment_summary.json)
  supplies the winner decision and uncertainty interval.

FN, TN, and accuracy are derived from the recorded counts and the common
population: `FN = 64 - TP`, `TN = 276 - FP`. To reproduce ranking metrics, load
the saved probabilities with `pd.read_csv(..., float_precision="round_trip")`.
Ordinary float parsing can slightly change score ties and thus the last digits
of ranking metrics. Round-trip parsing reproduces the recorded PR-AUC values.

Generated artifacts are ignored by Git and may be absent in a fresh checkout.
The tables preserve the inspected results. A future run should update the
population, inputs, operating rule, scores, and limitations together.

## Historical provisional-pipeline evaluation

The 2026-10-07 artifacts evaluate the tuned seven-feature models on the two
standard temporal folds. Threshold metrics use ensemble-average probabilities
at provisional thresholds 0.34 (Random Forest) and 0.24 (HistGradientBoosting).
Ranking metrics are computed independently for each seed and then averaged.

### Mean across folds

| Model | Mean recall | Mean precision | Mean F1 | Mean seed PR-AUC across folds | Worst-fold mean seed PR-AUC | Mean seed ROC-AUC across folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 47.38% | 34.89% | 0.3219 | 0.3520 | 0.2947 | 0.7289 |
| HistGradientBoosting | 63.44% | 28.29% | 0.3694 | 0.3269 | 0.3057 | 0.7358 |

These are equally weighted fold means, not metrics recomputed from all pooled
predictions. Mean F1 is the mean of each fold's F1, not F1 calculated from mean
precision and mean recall. Mean seed PR-AUC differs conceptually from PR-AUC
calculated once on ensemble-average scores. Do not directly compare these
summaries with the common-population ranking scores above.

### Per-fold operating results

| Model | Validation snapshot | Rows / churners | TP | FP | FN | Recall | Precision | F1 | Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 2013-01-01 | 343 / 25 | 19 | 56 | 6 | 76.00% | 25.33% | 0.3800 | 81.92% |
| HistGradientBoosting | 2013-01-01 | 343 / 25 | 20 | 74 | 5 | 80.00% | 21.28% | 0.3361 | 76.97% |
| Random Forest | 2013-04-01 | 340 / 64 | 12 | 15 | 52 | 18.75% | 44.44% | 0.2637 | 80.29% |
| HistGradientBoosting | 2013-04-01 | 340 / 64 | 30 | 55 | 34 | 46.88% | 35.29% | 0.4027 | 73.82% |

HistGradientBoosting catches more churners but generates more false alerts at
those historical thresholds in both periods. Both models' recall falls in April,
especially Random Forest. Mean scores alone conceal this variation; the
operating threshold and stability over time remain research decisions.

Evidence: [model summary](../results/provisional/development_model_summary.csv),
[fold metrics](../results/provisional/development_fold_metrics.csv), and
[development predictions](../results/provisional/development_predictions.csv).
Confusion counts were reconstructed from saved predictions and checked against
the reported operating metrics. See
[the evaluation script](../pipeline/11_provisional_training/02_evaluate_development_models.py)
for the seed and fold aggregation protocol.

## How to use these results

Choose an operating rule using development data and explicit retention capacity
or error costs. Compare recall together with precision, false alerts, missed
churners, and stability on matched populations. Freeze the complete methodology
before requesting protected final evaluation through
[Stage 12](../pipeline/12_final_evaluation/README.md). None of the results here
authorize or substitute for that final evaluation.

## Continued matched development research — 2026-10-16

These trials follow the current **fewer false alerts with no lower baseline
recall on the same splits** objective. All decisions additionally require the
predeclared practical gain, uncertainty and subgroup safeguards. None passes
its complete acceptance rule. No operational model/cutoff changes or October
outcome access occurred. New partitions of previously inspected development data
are stability evidence, not independent confirmation.

| Fixed candidate | Outer split seeds | Baseline mean recall / FP | Candidate mean recall / FP | Mean FP reduction | Decision |
| --- | --- | --- | --- | --- | --- |
| Add last-order line count | 42/78/88 | 91.51% / 520.67 | 91.35% / 522.67 | -2.00 (-0.38%) | Reject: recall and FP requirements fail |
| Add annual distinct products | 42/78/88 | 91.51% / 520.67 | 91.04% / 509.33 | 11.33 (2.18%) | Reject: recall, practical gain and uncertainty fail |
| Gap CV + overdue ratio, seven leaves, L2=1, no annual count | 301/919/1201 | 90.72% / 518.00 | 95.13% / 495.33 | 22.67 (4.38%) | Not accepted: practical gain and FP uncertainty fail |

Each row averages three repetition-pooled metrics. Each repeat evaluates the
same 1,375 cases once, with matched natural populations and grouped folds for
both arms; it does not supply 4,125 independent cases. Both arms maximize inner
OOF precision subject to recall >90%, with training-only weights, native missing
values and the original five-seed ensemble. Outer labels never set cutoffs.
The historical 0.06 rule is excluded from acceptance. The baseline itself falls
below 90% recall in new repeat 301 (89.62%); neither arm is retuned afterward.

The research practical default is >=5% mean FP reduction, alongside fewer FP
and recall >= baseline in **each** repeat. The activity/snapshot safeguard allows
at most five percentage points loss in mean recall; it passes in the three trials
above. Later learner/weight contrasts include subgroup failures, reported below.
These margins were frozen before fitting and are not user-supplied business
requirements. The cadence candidate's apparent benefit must not be relabelled
as acceptance by lowering the margin after seeing 4.38%.

| Candidate | Conditional recall-difference interval | Conditional FP-reduction interval | Interval definition |
| --- | --- | --- | --- |
| Last-order lines | -1.91 to +1.52 points | -14 to +10 | 2.5–97.5 percentiles |
| Annual products | -2.81 to +1.88 points | -2 to +25 | 1.25–98.75 percentiles for the two bounded trials/two endpoints |
| Cadence without annual count | +0.60 to +8.94 points | -1.33 to +49.67 | tails 0.05/(14×2), conservative for the inspected ablation family |

All use 5,000 paired reseller-cluster draws, seed 2026, moving reseller snapshots
and repeat predictions together. They condition on fitted models and cutoffs;
retraining/reselection, adaptive research and population-shift uncertainty remain.
The last trial passes recall and fewer-FP point checks in all three new repeats,
but its FP lower bound crosses zero. This does not establish that the candidate
has no benefit; it does not establish the frozen required improvement either.

Complete fold/repeat confusion counts, ranking near 90/95/100% recall, active/
inactive and snapshot results, annual-count/span/discount-missingness slices,
changed cases, cutoff distributions and seed spreads are in the respective
`report.md` and CSVs under `results/basket-breadth-20261016-serial/`,
`results/annual-breadth-20261016/` and `results/cadence-stability-20261016/`.
Independent no-fit verifiers reproduce 30 inner cutoffs, 15 grouped partitions,
600 recorded training identities and 36 overall metric rows per study. The two
original-split baselines reproduce all historical decisions (max score difference
1.11e-16). The last study has different training partitions, so old scores are
not direct fidelity targets.

See the frozen [basket](../comparisons/basket_breadth_protocol.md),
[annual breadth](../comparisons/annual_breadth_protocol.md) and
[cadence stability](../comparisons/cadence_stability_protocol.md) protocols,
[research journal](research_journal.md), and
[live source-availability findings](data_analysis.md#live-source-availability-follow-up--2026-10-16).


### Further fixed contrasts — 2026-10-17

The following comparisons use the same301/919/1201 partitions and ORIGINAL
balanced HGB primary baseline (mean518 FP,90.72% recall). All fail complete
acceptance; no original operating objective is weakened. Positive FP reduction
means fewer alerts; negative values mean additional false alerts.

| Procedure | Mean FP reduction | Main failed constraints |
| --- | --- | --- |
| Fixed50/50 original/cadence mean |22.00 (4.25%)|5% practical margin |
| RF10, balanced training | -25.33 |Recall, FP, practical, uncertainty, subgroup |
| Fixed50/50 original/RF10 mean | -4.00 |Recall, FP, practical, uncertainty, subgroup |
| L2 logistic10 | -302.67 |Recall, FP, practical, uncertainty |
| Unit-weight HGB10 | -9.33 |Recall, FP, practical, uncertainty |
| Unit-weight RF10 | -82.67 |FP, practical, uncertainty, subgroup |
| Cadence + direct SKU shares |17.00 (3.28%)|Per-repeat FP, practical, uncertainty |
| Fixed reseller-bootstrap SKU ensemble |5.00 (0.97%)|Per-repeat FP, practical, uncertainty |
| Cadence + five purchase-mix components | -15.67 |Per-repeat FP, practical, uncertainty |
| Symmetric-half raw cadence mean |23.33 (4.50%)|Practical, recall/FP uncertainty |
| Held-training-reference percentile cadence mean |51.33 (9.91%)|Per-repeat recall, recall uncertainty |
| Cadence + latest paid unit revenue |28.00 (5.41%)|FP-reduction uncertainty |

Each learned candidate has training-only preprocessing/weights, inner cutoffs
maximizing precision with recall>90%, matched evaluation rows and the frozen
same-repeat acceptance rule. Unit-weight RF preserves repeat recall but sharply
increases false alerts. Better curve/AP values in some studies do not rescue
failed operational requirements. Fixed blending is not followed by weight search.
Balanced RF is only a secondary reference in the weight study.

All failures, ranking curves, subgroup/sample-size reports, conditional paired
reseller uncertainty and independent audits are preserved under
results/fixed-blend-20261016/, results/forest-comparison-20261016/,
results/linear-comparison-20261016/ and results/weighting-comparison-20261016/.
The family-tail conventions and adaptive limitations are in their respective
comparisons/*_protocol.md files. A predeclared50,000-draw sign confirmation is
required only when every primary check passes in later trials; neither weight
candidate qualifies. No protected final evaluation or October outcome was used.


The fixed cadence+SKU representation (training-only frequency vocabulary, not
outcome-selected IDs) also fails acceptance on301/919/1201. TP/FP201/506,202/506,
199/491 versus original190/498,195/551,192/505: recall preserved, mean17 fewer FP
(3.28%), but eight extra FP in one repeat and both uncertainty checks fail.
Conditional intervals: recall[-1.31,+9.46]pp; FP reduction[-12.52,+47.67], tail.05/46.
Within-fold AP rises.5009→.5718 and descriptive high-recall burden falls; neither
rescues the frozen operational failure. All300 vocabulary/training records,
30 inner cutoffs and36 metric rows independently verify. Full report:
results/product-mix-comparison-20261017/report.md. No vocabulary retuning.


Both SKU follow-ups preserve original repeat recall and pass the activity/snapshot
safeguard, but fail complete acceptance. Bootstrap TP/FP199/507,204/520,205/512;
latent197/534,202/544,204/523. Neither receives50,000-draw confirmation or post-hoc
parameter tuning. Independent300-training-identity/30-cutoff/36-metric audits pass
for each. Reports: results/product-bagging-comparison-20261017/report.md and
results/latent-mix-comparison-20261017/report.md. The fixed direct-SKU vocabulary
captures little revenue, while5-PC compression captures only26.74% mean outer-
training variation; neither representation should be called complete information.

At the SAME baseline catch count, candidate outcome-derived tied-score ranking
points often reduce FP compared with baseline's own outcome-derived point. This
is **descriptive ranking evidence, not an operational cutoff or acceptance**.
DirectSKU improves12/15 outer folds (three worsen); cadence12 improve/one worsens/
two tie; bootstrap10/5; latent11/4. All480 inner/outer points reproduce by exhaustive
enumeration in results/matched-rank-burden-20261017/. Baseline cutoff slack is
reported separately; comparing candidate oracle FP only with baseline actual FP
would exaggerate separation. A fixed training-size sensitivity probe investigates
possible contributors without selecting new cutoffs or declaring a winner.


The symmetric-half alignment study shares600 component fits. Raw TP/FP193/470,
202/521,199/493; percentile189/454,199/490,196/456. Percentile alignment has a
positive conditional FP-reduction interval[21.50,82.67] but recall interval
[-2.83,+5.49]pp and loses one baseline-caught churner in301. The criterion is not
weakened for that loss. Raw passes recall/FP point checks but its4.50% gain and
both uncertainty bounds fail. Independent600-component/45-cutoff/36-metric-per-arm
audit passes; both full reports in results/crossfit-percentile-20261017/.
Reference percentiles are not calibrated churn probabilities. No reference-rule,
half-count, seed or cutoff tuning follows these outcomes.


### Latest fixed numeric-feature result — 2026-10-17

Cadence + latest paid unit revenue meets all point, practical, recall uncertainty
and subgroup requirements, but **does not pass complete acceptance**. Baseline
TP/FP190/498,195/551,192/505; candidate199/471,207/511,203/488. Mean recall90.72→95.75%,
FP518→490 (28 fewer,5.41%). Conditional paired reseller-cluster recall-difference
interval[+0.47,+10.38]pp; FP reduction[-2.51,+60.23], tail.05/56,5000 draws. The FP
lower bound crosseszero. No extra draws, threshold adjustment, altered tail or
parameter retry is used to rescue it; the frozen50,000-draw safeguard requires
ALL primary checks first and is therefore not triggered.

The secondary cadence reference is196/470,207/527,202/489; the added feature is
not uniformly better even on point counts versus that secondary procedure. The
primary comparator remains original balanced HGB10 with the same train-only
high-recall cutoff objective. Historical0.06 and October metrics are separate.
Independent no-fit audit passes15 grouped splits,30 cutoffs,36 overall metric
rows and all cached/new training hashes. Full report and failures:
results/cadence-unit-revenue-20261017/report.md. Full combined suite253 passed
in60.40s; Ruff and diff whitespace checks pass.

This promising developmental association remains unaccepted. Repeated seeds,
folds or bootstrap draws cannot create new independent reseller evidence or
remove adaptation from inspecting these same cases. Further independent
confirmation needs an appropriate untouched cohort with frozen data roles/method
and explicit authorization under the protected-evaluation rule; October cannot
serve that role. No additional source-backed remedy is currently justified by
the completed diagnoses. This is not proof that purchase histories contain no
other useful signal, nor permission for an indiscriminate search near a margin.

### Authorized latest-candidate October test — 2026-10-17

The user explicitly requested evaluation of the latest cadence/unit-revenue
candidate on October. Its13 features, seven-leaf HGB/L2=1, balanced weights and
five seeds remain fixed. A three-fold reseller-grouped development-only OOF
procedure selects cutoff0.18634605 (precision maximum with recall>90%). No test
labels select features, parameters or cutoffs. October492 cases include46 churners.

| Fixed procedure | TP | FP | FN | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical baseline0.06 | 43 | 108 | 3 | 93.48% | 28.48% |
| Latest cadence/unit-revenue candidate | 45 | 194 | 1 | 97.83% | 18.83% |
| Original10 with same development cutoff objective | 46 | 183 | 0 | 100.00% | 20.09% |

The candidate fails the false-alert reduction objective against both predeclared
references. Compared with historical0.06, it adds two catches and86 false alerts;
compared with the matched-policy baseline, it loses one catch and adds11 false
alerts. Recent purchasers account for the candidate miss and additional alerts.
All35 inactive churners and10 inactive continuing buyers are alerted by all arms.

Paired5000-reseller percentile95% intervals: recall difference versus historical
[0,+11.36]percentage points and FP reduction[-105,-67]; versus matched-policy
[-7.32,0]points and FP reduction[-29,+7]. They omit training,cutoff,source and
research adaptation uncertainty. Aggregate AP improves0.7013→0.7585 while recent
purchaser AP falls0.1553→0.0695; this does not establish operational improvement.

Models and predictions were saved before supplied outcomes were joined. October
was inspected historically; this is the authorized October test, not a newly
untouched cohort. No subsequent tuning or deployment. The no-fit verifier
reproduces40 training identities,two cutoffs,10 serialized-model predictions,nine
confusion rows and all paired intervals. Full evidence:
`results/unit-revenue-october-evaluation-20261017/completed_report/report.md`.
The prediction-stage manifest and archived as-run source preserve original code
identity; the current runner only removes an optional Markdown rendering dependency.


### Prepared development reporting and prospective safeguards — 2026-10-17

The shared assessment now writes operating_metrics.csv,paired_operating_metrics.csv
and worst_slices.csv. These preserve EVERY repeat/fold,snapshot,activity group and
snapshot/activity intersection. They report matched confusion counts,recall,
precision,FPR,alert rates and sample sizes; undefined rates stay undefined.
Harm flags compare to baseline and are descriptive unless a frozen prospective
policy explicitly makes them acceptance constraints. No historical metrics or
acceptance decisions change.

The no-fit review entrypoint and cutoff-transfer table are documented in
[the preparation protocol](../comparisons/development_research_protocol.md).
Inner pooled OOF,individual inner folds and outer validation use their saved
outer cutoff; component training row/reseller ranges are explicit. These tables
select no thresholds and cannot identify calibration or a causal failure mechanism.
Prospective per-repeat period/activity limits complement,not replace,global
recall preservation,FP reduction,practical improvement and paired uncertainty.
No candidate has been frozen or evaluated by this preparation.

## Development-only autonomous continuation — 2026-10-18

Ten further frozen comparisons completed on the same1375development rows,
212churn labels,422resellers and matched301/919/1201 grouped repeats. None passed
complete acceptance. The common training-only objective maximizes innerOOF
precision with recall>.90;the historical.06 and precision-floor policies remain
separate. Baseline mean recall90.72%,518false alerts per repeat.

| Procedure | Mean recall | Mean FP | FP reduction | Complete acceptance |
| --- | ---: | ---: | ---: | --- |
| Retained logistic reference calibration | 91.19% | 441.33 | 14.80% | Failed |
| Smooth additive cadence | 92.61% | 574.00 | -10.81% | Failed |
| Inverse reseller-history weighting | 90.41% | 549.00 | -5.98% | Failed |
| Fixed activity experts | 90.41% | 497.00 | 4.05% | Failed |
| Fixed complementary calibrated/expert mean | 91.19% | 448.67 | 13.38% | Failed |
| Equal period/class weighting | 90.41% | 538.67 | -3.99% | Failed |
| RBF cadence margin learner | 92.30% | 659.67 | -27.35% | Failed |
| Earliest-quarter historical order share | 91.67% | 524.00 | -1.16% | Failed |
| Training-guarded component with baseline fallback | 90.72% | 518.00 | 0.00% | Failed |
| Baseline alert veto below caught-inner-positive floor | 90.25% | 476.33 | 8.04% | Failed |

Values average repeat-level pooled metrics;these are dependent development
observations,not seven independent cohorts. Positive FP reduction means fewer.
Favorable means for calibration/blending still lose baseline catches in301 and
violate frozen period/activity/intersection safeguards. No50000draw confirmation
was permitted for a failed primary. Prospective alpha-spending and conditional
cluster intervals do not cover all prior adaptive research,retraining or shifts.

The [continuation report](../results/development-continuation-20261018/report.md)
owns consolidated counts,intervals,identities and limitations;each source study
retains every fold,period,activity/intersection and undefined/sparse flag. The first
seven complete acceptance decisions independently reproduce from case-level
predictions and paired5000reseller-cluster draws. Frozen procedures are in
comparisons/{retained_calibration,smooth_cadence,reseller_weighting,activity_experts,
continuation_blend,period_weighting,rbf_cadence}_protocol.md. Runtime interruption
and all-missing expert-column handling amendments are preserved. No operational
model was changed,no final evaluation ran,and no candidate is promoted.


Cycles8–10 preserve their separately frozen protocols. Quarter-share fails FP;
guarded selection falls back in all15folds;veto reduces FP but loses baseline
recall and fails October/activity limits. Each complete decision independently
reproduces. Exact empirical bootstrap sign counting is an additional prospective
cycle10 gate after all5k/50kgates pass;it changes no historical decision and provides
no adaptive/future guarantee. Failedprimary correctly prevents extra confirmation.
The [wrap-up report](../results/development-wrapup-20261018/report.md) and safe
checkpoint own the latest status and reproducible source/identity bundle.


## Authorized veto October evaluation — 2026-10-18

The user authorized evaluating the latest positive-floor veto. October was already
inspected, so this is descriptive reuse. Saved predictions preceded joining supplied
outcomes; no test tuning. On492resellers/46churners:

| Procedure | Recall | Precision | TP | FP | FN | TN | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original10, matched development cutoff |100.00%|20.09%|46|183|0|263|229|
| Positive-floor veto |100.00%|21.20%|46|171|0|275|217|
| Historical cutoff0.06 |93.48%|28.48%|43|108|3|338|151|

The veto removes12FP(6.56%) and losesno catches against its matched comparator;
95%conditional reseller-bootstrap FPreduction interval[6,19],recallchange[0,0].
Historical0.06 retains betterprecision withlowerrecall. Full development acceptance
remainsFAILED;no deployment or independentvalidation claim. All nine subgroup
metric rows,source/checkpoint identities,reporting recovery and independent audits
are in [the report](../results/positive-floor-october-evaluation-20261018/summary.md).


## Candidate-specific threshold follow-up — 2026-10-18

User requested selecting the candidate's own development threshold before October
evaluation. Removed baseline-TP and baseline-cutoff constraints; maximized pooled
OOF precision subject to recall strictly above 90%, ties recall then higher cutoff.
Independent exhaustive tied-score enumeration agrees with the existing selector.
It selects the SAME cutoff,0.0067626906041783315: development recall90.09%,precision
28.72%,191TP/474FP/21FN/689TN. No numeric threshold difference or changed test
predictions. October remains46TP/171FP/0FN/275TN,100%recall/21.20%precision.
No refit, floor revision or test-based selection. Predictions checkpointed before
supplied outcomes; manifests verified. Descriptive reuse of inspected October;
full development acceptance still failed. Report:
results/positive-floor-own-threshold-october-20261018/evaluation/report.md.


## Authorized precision-floor October sweep — 2026-10-18

User requested minimum development precision20%,30%,40%,... then final evaluation.
Fixed model/veto floor; no new fits. Frozen targets20–100% in10point steps; maximize
development OOF recall with precision >=target, ties higherprecision then cutoff.
No90%recall floor in this policy. All nine cutoffs and test predictions saved
before supplied outcomes. Independent exhaustive tied-cutoff selection and27group
confusion recounts pass;two synthetic tests pass. Target30% gives test45TP/154FP,
recall97.83%,precision22.61%;40% gives40TP/60FP,86.96%,40.00%;50% gives31TP/23FP,
67.39%,57.41%. Targets80–100% select one development alert and no test alerts;
test precision undefined, explicitly flagged. No test-winner selection or deployment.
October previously inspected; descriptive reuse. Full table and undefined-rate
correction:results/precision-floor-october-20261018/report.md. Original outputs
remain immutable;audited_comparison.csv provides undefined precision display.


## Authorized all-candidate precision-floor evaluation — 2026-10-18

User requested the same precision-target evaluation for other candidates. Frozen
scope: ten recent procedures plus two earlier cadence candidates and original
baseline,13 total. Development OOF cutoffs maximize recall subject to precision
>=20%,30%,...,100%;ties higherprecision then cutoff;no recallfloor. All117targets
frozen before outcome stage. 92 feasible and25 infeasible,all reported. Same492
October resellers/46churners,inspected previously;descriptive reuse,not independent
validation. No test winner selection,model tuning,deployment,merge or push.

At40%developmentprecision target,testprecision/recall/FP: original40.00/86.96/60;
retainedcalibration38.18/91.30/68;smooth52.86/80.43/33;reseller42.71/89.13/55;
experts43.30/91.30/55;blend41.58/91.30/59;period45.16/91.30/51;RBF43.82/84.78/50;
earlyquarter39.62/91.30/64;guarded=original;veto40.00/86.96/60;oldercadence35.40/
86.96/73;unitrevenue34.26/80.43/71. These are separate predeclared policy outcomes,
not a test-selected winning method. Full20–100%grid and counts in the report.

Independent audit:117cutoff choices,128logicalfitidentities,7fullmodelbundles,
276overall/activity metricrows all reproduce. Guarded procedure selects baseline
fallback;its duplicate results are intentional. Legacycadence runtime/Windows
manifest incompatibilities handled via verified originalspec and fixed refit.
Interruptedpreparation attempts preserved;fullmodels cached,75losttransientinner
HGB predictions recovered by exactidentity-matched fits. Finalroot:
results/all-candidate-precision-october-20261018-04/report.md. All frozen
DEVELOPMENT acceptance failures remain;no independent final-validation claim.


### Complete October precision-target table, including personalized recency

Each cell is **October precision / recall (%)**. Cutoffs maximize pooled
three-fold development OOF recall subject to the column's minimum precision;
ties prefer higher precision, then higher cutoff. No 90% recall floor applies.
Reseller-grouped folds use seed 78. Scores tied at a cutoff stay together.
All 492 October cases and 46 churners remain. An infeasible development target
has no test policy; undefined precision must not be displayed as successful.

| Procedure | 20% target | 30% target | 40% target | 50% target | 60% target | 70% target |
|---|---|---|---|---|---|---|
| Original baseline | 12.92 / 100.00 | 22.84 / 97.83 | 40.00 / 86.96 | 57.41 / 67.39 | 70.00 / 45.65 | 78.26 / 39.13 |
| Retained calibration | 13.07 / 100.00 | 21.03 / 97.83 | 38.18 / 91.30 | 44.44 / 78.26 | 63.46 / 71.74 | 83.33 / 65.22 |
| Smooth additive cadence | 13.58 / 95.65 | 30.07 / 93.48 | 52.86 / 80.43 | 81.58 / 67.39 | 88.89 / 52.17 | Infeasible |
| Reseller weighting | 11.25 / 100.00 | 24.18 / 95.65 | 42.71 / 89.13 | 56.67 / 73.91 | 69.70 / 50.00 | 76.92 / 43.48 |
| Activity experts | 13.11 / 100.00 | 25.28 / 97.83 | 43.30 / 91.30 | 58.62 / 73.91 | 78.57 / 47.83 | 94.74 / 39.13 |
| Complementary blend | 13.03 / 100.00 | 21.80 / 100.00 | 41.58 / 91.30 | 55.88 / 82.61 | 73.68 / 60.87 | 92.59 / 54.35 |
| Period/class weighting | 11.92 / 100.00 | 23.32 / 97.83 | 45.16 / 91.30 | 56.14 / 69.57 | 77.78 / 45.65 | 93.33 / 30.43 |
| RBF cadence | 13.11 / 100.00 | 27.85 / 95.65 | 43.82 / 84.78 | 69.57 / 69.57 | 72.00 / 39.13 | 0.00 / 0.00 |
| Early-quarter share | 11.86 / 100.00 | 20.09 / 100.00 | 39.62 / 91.30 | 58.93 / 71.74 | 79.31 / 50.00 | 81.82 / 39.13 |
| Guarded component¹ | 12.92 / 100.00 | 22.84 / 97.83 | 40.00 / 86.96 | 57.41 / 67.39 | 70.00 / 45.65 | 78.26 / 39.13 |
| Positive-floor veto | 16.08 / 100.00 | 22.61 / 97.83 | 40.00 / 86.96 | 57.41 / 67.39 | 70.00 / 45.65 | 78.26 / 39.13 |
| Earlier cadence/regularization | 12.23 / 100.00 | 17.97 / 100.00 | 35.40 / 86.96 | 43.75 / 76.09 | 69.39 / 73.91 | 90.00 / 58.70 |
| Earlier cadence/unit revenue | 12.30 / 100.00 | 18.00 / 97.83 | 34.26 / 80.43 | 43.75 / 76.09 | 63.46 / 71.74 | 83.33 / 65.22 |
| Period/class + personalized recency | 11.41 / 100.00 | 19.91 / 100.00 | 38.74 / 93.48 | 50.77 / 71.74 | 77.14 / 58.70 | 86.36 / 41.30 |


¹ Guarded component selected the baseline fallback. Its duplicate row is
intentional. RBF at 70% has false alerts and no caught churners, so its 0/0
entry means zero precision and zero recall, not undefined no-alert precision.
The older sweep also reports 80%–100% targets, including infeasible and undefined
cases, in its original artifact; this table shows the requested 20%–70% range.

The first 13 rows are preserved from the original sweep. The new row uses the
same selection policy with the fixed period/class + overdue-ratio procedure.
It required 15 grouped OOF and five full-development HGB fits; all models and
predictions were saved. Cutoffs and unlabeled decisions were checkpointed before
joining supplied outcomes. No test labels selected cutoffs or revised this model.
These results reuse inspected October; they are not independent confirmation.

| Personalized recency target | TP | FP | FN | TN | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20% | 46 | 357 | 0 | 89 | 403 |
| 30% | 46 | 185 | 0 | 261 | 231 |
| 40% | 43 | 68 | 3 | 378 | 111 |
| 50% | 33 | 32 | 13 | 414 | 65 |
| 60% | 27 | 8 | 19 | 438 | 35 |
| 70% | 19 | 3 | 27 | 443 | 22 |

At 40%, period/class alone gives 42TP/51FP/4FN/395TN: personalized recency
adds one TP and 17 FP. This is a worse false-alert trade-off for the stated
objective at that point; it does not mean the feature is useless or worse at
every target. At 60%, its recall is 58.70% versus 45.65%, with precision
77.14% versus 77.78%. No company precision minimum or outreach capacity has
been established, so none of these columns is an approved business policy.

[Full updated report](../results/personalized-recency-october-20261018/evaluation/report.md),
[all overall/activity confusion counts](../results/personalized-recency-october-20261018/evaluation/combined_metrics.csv),
[development cutoffs](../results/personalized-recency-october-20261018/development_selections.csv),
and [independent verification](../results/personalized-recency-october-20261018/independent_verification.json)
retain the evidence. Earlier raw reports and manifests were not rewritten.


## Development ranking-information research — 2026-10-18

Four fixed single-feature additions to period/class-weighted HGB10 used the
same 1,375 development cases, 212 churn cases and 422 resellers. Repeats
301/919/1201 each have five outer folds and three inner folds (seed 78), with
reseller isolation at both levels. Cutoffs maximize inner OOF recall subject to
precision >=40%, with the same tie rules in both arms. No October inputs were
used in this development cycle. Parameters, five model seeds, native missingness
and period/class weights stayed fixed. Identical seed predictions do not count
as independent robustness evidence.

| Added predictor | Mean recall | Mean TP | Mean FP | Recent AP gain, points | Descriptive 95% AP-gain interval, points | Complete acceptance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Reference: period/class ten features | 63.84% | 135.33 | 217.67 | — | — | Comparator |
| Annual distinct products | 64.78% | 137.33 | 214.33 | +0.75 | [-0.99, 2.67] | Failed |
| Distinct products / annual orders | 67.61% | 143.33 | 227.67 | +2.12 | [-0.33, 4.77] | Failed |
| Personalized recency | 67.45% | 143.00 | 213.67 | +3.27 | [0.92, 5.59] | Failed |
| Annual accepted-order count control | 64.31% | 136.33 | 220.67 | +1.77 | [-0.20, 3.78] | Failed |

These are means of dependent repeat-level pooled operating counts, not unique
customer totals. Recent average precision (AP) averages within-fold ranking
metrics; it is not precision among the chosen alerts. Personalized recency
raises recent AP from 27.64% to 30.92%, with positive point gains in all repeats.
Its mean recent AUC gain is 3.20 points, descriptive 95% interval [1.95, 4.54].

| Repeat | Reference TP / FP | New TP | New FP | New FN | New TN | New recall | New precision | New alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 301 | 134 / 214 | 146 | 210 | 66 | 953 | 68.87% | 41.01% | 356 |
| 919 | 129 / 199 | 136 | 203 | 76 | 960 | 64.15% | 40.12% | 339 |
| 1201 | 143 / 240 | 147 | 228 | 65 | 935 | 69.34% | 39.20% | 375 |
| Mean | 135.33 / 217.67 | 143.00 | 213.67 | 69.00 | 949.33 | 67.45% | 40.11% | 356.67 |

Personalized recency gains 7.67 TP and removes four FP on average, but repeat 919
adds four FP. Its 1.84% mean FP reduction is below the frozen 5% requirement;
period/activity/intersection and operating uncertainty gates also fail. Annual
breadth loses recall and its AP gain reverses in one repeat. Density adds ten FP
on average; annual count adds three and loses recall in one repeat. Their AP
intervals include zero. Density minus count gives only a 0.34-point AP increment,
with descriptive paired interval [-2.53, 3.41]: product variety is not established
as the unique source of the density gain.

All four acceptance failures stand. Paired reseller bootstrap keeps snapshots
and repeat predictions together; these conditional intervals omit retraining,
adaptive search, calendar dependence and future shifts. Ranking intervals are
descriptive, not corrected evidence of a winner. Frozen cycle 11–14 spending and
operating safeguards were not relaxed; failed primaries received no extra-draw
rescue. Every fold, period, activity group and intersection is in the source
studies rather than hidden by the averages above.

The separate full-development three-fold OOF selector gives personalized
recency 65.09% recall and 40.35% precision at its 40% target (138TP/204FP).
That selector is used to choose the full-fit cutoff; it is not the nested outer
67.45%/40.11% estimate above. Period/class alone's corresponding selector gives
63.21%/41.49% (134TP/189FP). Neither should be substituted for the October table.

[Consolidated development report](../results/development-ranking-information-20261018/report.md),
[every repeat's counts](../results/development-ranking-information-20261018/repeat_metrics.csv),
and [verification](../results/development-ranking-information-20261018/verification.json)
record source, code, protocol, model, weight and runtime identities. All 1,200
saved estimators, 120 selected cutoffs and 2,160 operating rows were independently
reconstructed; full 5,000-draw operating bootstraps were checked for every arm.
Independent ranking-bootstrap checks cover 17 draws per arm, not all 5,000.
The later authorized personalized-recency October evaluation appears above and
does not retroactively change this development-only cycle's decision.

### What the missed-case diagnosis establishes

Under the period/class 40%-target development OOF policy, 67 recent-purchase
churn cases were missed (66 resellers). Of these, 47 have an observed selected
warning, 15 have none among defined comparisons, and five lack enough comparison
history. Warnings are also common among continuing buyers: smaller last orders
occur in 64.5% of defined missed cases versus 67.3% of continuing cases; revenue
decline in 43.4% versus 63.2%; longer last gaps in 39.2% versus 39.5%.
Same-period matching on original features does not establish these binary flags
as separators. Absence of a useful flag is not proof that churners show no signs.
Continuous features may add conditional information even when a binary flag fails.
[The development warning audit](../results/hard-recent-warning-audit-20261018-02/report.md)
retains missingness, all controls and sensitivity results.

Period/class alone's 40%-target selector catches 63.21% in development but
91.30% in October. October contains 35/46 churn cases in the no-recent-purchase
group, versus 91/212 in development. Applying development subgroup recalls to
October's churn mix gives a descriptive 77.56%, so composition explains part,
not all, of the difference. Within-group performance differs too. This is not
proof that October customers are universally 'easy' or that every missed buyer
has no available warning. Inner-to-outer cutoff transfer is observed; its cause
and whether it explains repeat 919 remain unresolved. The next useful research
step, if requested, is a saved-score diagnosis before another fit or policy.

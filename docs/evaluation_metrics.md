# Evaluation metrics and latest results

Updated on 2026-10-11 against the saved authorized final evaluation and earlier
development artifacts. The latest evaluation tests the frozen ten-feature
boosting ensemble on October 2013, with grouped-development rules fixed before
outcome access. Earlier seven-versus-ten validation experiments retain their
own protocols below.

The historical provisional pipeline's latest available evaluation files are dated
2026-10-07. Its results are recorded separately below because its models and
evaluation summaries differ. See [project_context.md](project_context.md) for
data roles and [feature_analysis.md](feature_analysis.md) for model inputs.

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

## Current development selection reports

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

## Latest completed experiment: seven versus ten features

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
label availability. Temporal remains primary for the stated future-churn goal;
both methods are retained for development reporting.

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

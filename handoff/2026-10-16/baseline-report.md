# Baseline development error diagnosis

This report describes the original ten-feature model's saved held-out predictions.
No models were fitted, new cutoffs selected, candidates compared, or October
outcomes accessed. All four approved development snapshots are included. The
population is 1,375 account/snapshot cases from 422 resellers, with 212 churn cases.
Each case has three dependent held-out predictions; this is not 4,125 independent cases.

## Main observations


- **The recent-purchase group remains difficult.** Under inner-selected cutoffs,
  mean repeat recall is 85.12%, precision 18.38%, and
  false-positive rate 41.61%. Accounts without a recent
  purchase have recall 100.00%, but false-positive rate
  100.00%: all continuing buyers in that group are alerted.
- **Many errors persist across fits.** There are 416 account/snapshot
  cases falsely alerted in all three repetitions (353 recently
  purchasing), and 11 churn cases missed in all three. These are
  distinct cases, not counts of repeated predictions; see account_stability.csv.
- **Sparse history alone does not explain the misses.** 46 of
  54 missed-churn events across repetitions have three or more annual
  orders. This is a dependent event count, not independent churners. Group sizes
  and rates are needed to interpret it; more orders do not guarantee separability.
- **Operating policy has a substantial cost.** Compared with 0.06 on identical
  scores, the saved high-recall policy catches 30.00 more churn cases per
  repetition on average but adds 187.00 false alerts. No ranking changed.
- **Stability and snapshot variation matter.** 383 of 1,375 cases move
  at least 20 rank-percentile points across fits. The highest snapshot-level
  mean false-positive rate under the high-recall policy is
  59.66% in 2013-04-01. These grouped
  snapshot slices are descriptive, not future-period validation.


## Verified operating results

The fixed 0.06 rule is a retrospective historical reference. The inner-selected
rule uses the saved training-only policy maximizing precision with recall >90%.
It is not a newly selected cutoff or the original 30%-precision selection policy.

| policy | repeat | tp | fp | fn | recall | precision | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_006 | 42 | 160 | 324 | 52 | 0.7547 | 0.3306 | 0.2786 |
| fixed_006 | 78 | 167 | 337 | 45 | 0.7877 | 0.3313 | 0.2898 |
| fixed_006 | 88 | 165 | 340 | 47 | 0.7783 | 0.3267 | 0.2923 |
| inner_selected | 42 | 195 | 530 | 17 | 0.9198 | 0.2690 | 0.4557 |
| inner_selected | 78 | 192 | 497 | 20 | 0.9057 | 0.2787 | 0.4273 |
| inner_selected | 88 | 195 | 535 | 17 | 0.9198 | 0.2671 | 0.4600 |

## Recent purchasing activity

Recent purchase means at least one accepted order during the preceding three
months. No recent purchase still means eligible under the six-month rule.

| policy | repeat | group | rows | churners | tp | fp | fn | recall | precision | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_006 | 42 | recent_purchase | 1221 | 121 | 71 | 263 | 50 | 0.5868 | 0.2126 | 0.2391 |
| fixed_006 | 42 | no_recent_purchase | 154 | 91 | 89 | 61 | 2 | 0.9780 | 0.5933 | 0.9683 |
| fixed_006 | 78 | recent_purchase | 1221 | 121 | 77 | 276 | 44 | 0.6364 | 0.2181 | 0.2509 |
| fixed_006 | 78 | no_recent_purchase | 154 | 91 | 90 | 61 | 1 | 0.9890 | 0.5960 | 0.9683 |
| fixed_006 | 88 | recent_purchase | 1221 | 121 | 76 | 278 | 45 | 0.6281 | 0.2147 | 0.2527 |
| fixed_006 | 88 | no_recent_purchase | 154 | 91 | 89 | 62 | 2 | 0.9780 | 0.5894 | 0.9841 |
| inner_selected | 42 | recent_purchase | 1221 | 121 | 104 | 467 | 17 | 0.8595 | 0.1821 | 0.4245 |
| inner_selected | 42 | no_recent_purchase | 154 | 91 | 91 | 63 | 0 | 1.0000 | 0.5909 | 1.0000 |
| inner_selected | 78 | recent_purchase | 1221 | 121 | 101 | 434 | 20 | 0.8347 | 0.1888 | 0.3945 |
| inner_selected | 78 | no_recent_purchase | 154 | 91 | 91 | 63 | 0 | 1.0000 | 0.5909 | 1.0000 |
| inner_selected | 88 | recent_purchase | 1221 | 121 | 104 | 472 | 17 | 0.8595 | 0.1806 | 0.4291 |
| inner_selected | 88 | no_recent_purchase | 154 | 91 | 91 | 63 | 0 | 1.0000 | 0.5909 | 1.0000 |

## Other predefined profiles

The following values are arithmetic means of repeat-level metrics, not an
independent validation set. Complete per-repeat/per-fold counts and discount
availability slices are in subgroup_metrics.csv (`subgroup_metrics.csv` in the restored results).
Observed history span is last minus first purchase inside the twelve-month
window, not true relationship tenure. A one-order history has span zero.

| policy | slice | group | recall | precision | false_positive_rate | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_006 | annual_orders | 1 | 0.9259 | 0.2518 | 0.3318 | 74.3333 | 2.0000 |
| fixed_006 | annual_orders | 2 | 0.8000 | 0.2927 | 0.2892 | 67.6667 | 7.0000 |
| fixed_006 | annual_orders | 3+ | 0.7400 | 0.3668 | 0.2719 | 191.6667 | 39.0000 |
| fixed_006 | history_span | 0-90 days | 0.9062 | 0.2781 | 0.3304 | 75.3333 | 3.0000 |
| fixed_006 | history_span | 181+ days | 0.7460 | 0.3758 | 0.2846 | 208.3333 | 42.6667 |
| fixed_006 | history_span | 91-180 days | 0.8056 | 0.1620 | 0.2463 | 50.0000 | 2.3333 |
| fixed_006 | snapshot_slice | 2012-07-01 | 0.8026 | 0.4477 | 0.3013 | 75.3333 | 15.0000 |
| fixed_006 | snapshot_slice | 2012-10-01 | 0.9149 | 0.3316 | 0.2717 | 86.6667 | 4.0000 |
| fixed_006 | snapshot_slice | 2013-01-01 | 0.7467 | 0.1963 | 0.2411 | 76.6667 | 6.3333 |
| fixed_006 | snapshot_slice | 2013-04-01 | 0.6458 | 0.3030 | 0.3442 | 95.0000 | 22.6667 |
| inner_selected | annual_orders | 1 | 1.0000 | 0.2136 | 0.4449 | 99.6667 | 0.0000 |
| inner_selected | annual_orders | 2 | 0.9238 | 0.2402 | 0.4373 | 102.3333 | 2.6667 |
| inner_selected | annual_orders | 3+ | 0.8978 | 0.2971 | 0.4520 | 318.6667 | 15.3333 |
| inner_selected | history_span | 0-90 days | 0.9792 | 0.2331 | 0.4532 | 103.3333 | 0.6667 |
| inner_selected | history_span | 181+ days | 0.8968 | 0.3071 | 0.4645 | 340.0000 | 17.3333 |
| inner_selected | history_span | 91-180 days | 1.0000 | 0.1345 | 0.3810 | 77.3333 | 0.0000 |
| inner_selected | snapshot_slice | 2012-07-01 | 0.9342 | 0.4007 | 0.4253 | 106.3333 | 5.0000 |
| inner_selected | snapshot_slice | 2012-10-01 | 0.9929 | 0.2802 | 0.3762 | 120.0000 | 0.3333 |
| inner_selected | snapshot_slice | 2013-01-01 | 0.8800 | 0.1452 | 0.4078 | 129.6667 | 3.0000 |
| inner_selected | snapshot_slice | 2013-04-01 | 0.8490 | 0.2481 | 0.5966 | 164.6667 | 9.6667 |

## Ranking around high recall

First, mean within-fold ranking metrics (folds equally weighted; not independent):

| slice | group | average_precision | roc_auc |
| --- | --- | --- | --- |
| activity | no_recent_purchase | 0.7372 | 0.6593 |
| activity | recent_purchase | 0.2962 | 0.7752 |
| overall | all | 0.5108 | 0.8315 |

Below are descriptive mixed-model pooled curves at predeclared recall levels.
Held-out labels locate these points; they MUST NOT become deployment cutoffs.
Tied scores enter together, so achieved recall can exceed the target. Each
fold's separate curve and burden are retained in the CSVs. Policies share
identical scores and therefore identical ranking.

| repeat | subgroup | target_recall | achieved_recall | tp | fp | precision |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | overall | 0.9000 | 0.9009 | 191 | 493 | 0.2792 |
| 42 | overall | 0.9500 | 0.9528 | 202 | 640 | 0.2399 |
| 42 | overall | 1.0000 | 1.0000 | 212 | 1125 | 0.1586 |
| 42 | recent_purchase | 0.9000 | 0.9008 | 109 | 565 | 0.1617 |
| 42 | recent_purchase | 0.9500 | 0.9504 | 115 | 712 | 0.1391 |
| 42 | recent_purchase | 1.0000 | 1.0000 | 121 | 1062 | 0.1023 |
| 78 | overall | 0.9000 | 0.9009 | 191 | 478 | 0.2855 |
| 78 | overall | 0.9500 | 0.9528 | 202 | 694 | 0.2254 |
| 78 | overall | 1.0000 | 1.0000 | 212 | 1118 | 0.1594 |
| 78 | recent_purchase | 0.9000 | 0.9008 | 109 | 521 | 0.1730 |
| 78 | recent_purchase | 0.9500 | 0.9504 | 115 | 669 | 0.1467 |
| 78 | recent_purchase | 1.0000 | 1.0000 | 121 | 1055 | 0.1029 |
| 88 | overall | 0.9000 | 0.9009 | 191 | 491 | 0.2801 |
| 88 | overall | 0.9500 | 0.9528 | 202 | 661 | 0.2341 |
| 88 | overall | 1.0000 | 1.0000 | 212 | 1080 | 0.1641 |
| 88 | recent_purchase | 0.9000 | 0.9008 | 109 | 578 | 0.1587 |
| 88 | recent_purchase | 0.9500 | 0.9504 | 115 | 651 | 0.1501 |
| 88 | recent_purchase | 1.0000 | 1.0000 | 121 | 1017 | 0.1063 |

## Persistent versus unstable errors

Patterns refer to the same reseller/snapshot across three repetitions. Consistent
means wrong in 3/3; recurring means 2/3; occasional means 1/3. Counts across
snapshots of one reseller are dependent. Rank percentiles use midranks within
the complete held-out fold; ties are not separated by account ID.

| policy | error_pattern | account_snapshots | resellers |
| --- | --- | --- | --- |
| fixed_006 | consistent_error | 269 | 177 |
| fixed_006 | no_error | 880 | 374 |
| fixed_006 | occasional_error | 114 | 103 |
| fixed_006 | recurring_error | 112 | 93 |
| inner_selected | consistent_error | 427 | 219 |
| inner_selected | no_error | 721 | 335 |
| inner_selected | occasional_error | 119 | 94 |
| inner_selected | recurring_error | 108 | 95 |

Consistent errors by activity and actual outcome (churn=0 means false alerts):

| policy | activity | churn | account_snapshots |
| --- | --- | --- | --- |
| fixed_006 | no_recent_purchase | 0 | 59 |
| fixed_006 | recent_purchase | 0 | 180 |
| fixed_006 | recent_purchase | 1 | 30 |
| inner_selected | no_recent_purchase | 0 | 63 |
| inner_selected | recent_purchase | 0 | 353 |
| inner_selected | recent_purchase | 1 | 11 |

Rank instability means a range of at least 20 percentile points across fits.
Decision instability means alerts in some but not all repetitions. These are
descriptive flags, not tests of statistical significance. Each repetition also
changes the held-out comparison pool; percentile variation therefore combines
model/refit changes with fold composition and cannot isolate model instability.

| policy | rank_unstable | decision_unstable |
| --- | --- | --- |
| fixed_006 | 383 | 226 |
| inner_selected | 383 | 227 |

Saved outer-fold cutoffs:

| policy | min | median | max |
| --- | --- | --- | --- |
| fixed_006 | 0.0600 | 0.0600 | 0.0600 |
| inner_selected | 0.0034 | 0.0069 | 0.0175 |

Changing a cutoff explains decision differences on identical scores. This audit
cannot isolate why scores differ across refits or prove that a new cutoff will
transfer to a full-data model. A further controlled stability experiment would
be required for that question.

## Inspection files and plots

- Account-level diagnostics (`account_diagnostics.csv` in the restored results): features, profiles, ranks,
  both decisions, and distance to the alert boundary. Near-boundary means within
  five percentile points. High-ranked false alerts are flagged in the top 25%;
  low-ranked missed churners in the bottom 25%. These are review categories.
- Account stability (`account_stability.csv` in the restored results): recurrence and policy agreement.
- Rank-band metrics (`rank_band_metrics.csv` in the restored results) and ranking curves (`ranking_curves.csv` in the restored results).
- Descriptive high-recall burden (`high_recall_burden.csv` in the restored results).
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

108 saved baseline metric rows were
independently reproduced; all 15 outer splits and saved inner cutoff provenance
were audited. Input and output identities are in manifest.json. This checks
saved predictions and metadata, not estimator refits or an independent fresh
source-completeness audit. Small samples (rows <30, churners <10, or non-churners
<10) are flagged; empty or undefined rates remain missing. No confidence intervals
are claimed. Grouped snapshots describe unseen identities in mixed historical
periods, not future-period generalization. Development has been inspected before;
these findings are exploratory and cannot establish an independent improvement.

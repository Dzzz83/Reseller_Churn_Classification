# Churn debugging and feature experiments

This document owns the detailed debugging evidence and preserves each completed
experiment under its original configuration. The main feature catalog is in
[feature_analysis.md](feature_analysis.md); the current result summary is in
[evaluation_metrics.md](evaluation_metrics.md#nested-grouped-revenue-comparison--2026-10-15).

Completed investigations, in order:

1. Original April RFM missed-churn diagnosis below.
2. [RFM versus seven features](#experiment-rfm-versus-seven-features).
3. [CompanyX data discovery and 11 candidate features](#companyx-feature-discovery-and-usefulness-experiment--2026-10-10).
4. [Seven versus ten across three models and both validations](#seven-versus-ten-features-across-models-and-validation-methods--2026-10-10).
5. [Authorized final testing evaluation](#final-testing-evaluation--2026-10-11).
6. [Additional RF7 evaluation on the inspected October cohort](evaluation_metrics.md#additional-rf7-evaluation-on-the-inspected-final-cohort--2026-10-14), completed on 2026-10-14.
7. [SubTotal versus TotalDue grouped comparison](#completed-grouped-revenue-comparison--2026-10-14).
8. [Retrospective October revenue comparison](#authorized-retrospective-october-extension--2026-10-15).
9. [Repeated nested grouped revenue study](#completed-nested-grouped-study--2026-10-15).
10. [Held-out SHAP explanations](#completed-held-out-shap-analysis--2026-10-16).

The development comparison completed 154 fits with 39 focused tests passing.
The subsequent authorized final evaluation refitted five boosting models,
with 47 focused tests passing. The main pipeline feature registry is unchanged.
The user froze grouped-development thresholds for this final test, with 30%
primary and 20%, 40%, 50% secondary. The October outcomes are now inspected
and must not be called an untouched test in further research. No production
model promotion has been performed. Earlier sections preserve their original
development settings, results, and verification records. The latest nested
study excludes October data and does not establish reliable TotalDue dominance.

## Original April 2013 missed-churn diagnosis

Completed on 2026-10-10 using existing development predictions. The first
diagnostic question was: **did missed churners show a warning in their purchase
history before the prediction date, and did that warning distinguish them from
resellers who continued buying?**

The original finding is that this RFM model misses recently active churners. Some
have large spending declines that its three inputs do not describe, but declines
also occur among retained accounts. Other misses have purchase histories very
similar to retained accounts. This supports investigating richer signals; it
does not establish that any proposed feature will improve recall at acceptable
precision.

## Scope and operating point

| Item | Value |
| --- | --- |
| Source run | `codex-luna-verification-20261010-1` |
| Validation snapshot | 2013-04-01, saved `fold_2` predictions |
| Fold training snapshots | 2012-07-01 and 2012-10-01 |
| Candidate | `logistic_regression__rfm__balanced_weights__3aed3be011` |
| Inputs | `recency_days`, `n_orders_12m`, `revenue_12m` |
| Settings | Balanced class weights, C=1.0, max_iter=2000, seed 42 |
| Alert rule | Saved score >= 0.77 |
| Selection scenario | Minimum precision 0.30 in each development fold |
| Historical window examined | [2012-04-01, 2013-04-01) |

The 30% floor is a sensitivity scenario, not an approved business requirement.
The source search tested only RFM, nine configurations, one seed, and no tuning.
Its validation labels already participated in candidate/threshold selection
and earlier inspection. This report is descriptive development debugging,
not an independent estimate of future performance.

The source [selection manifest](../results/codex-luna-verification-20261010-1/development/selection/precision_03_0.3.json)
defines the candidate and threshold. Predictions come from
[validation_predictions.csv](../results/codex-luna-verification-20261010-1/development/validation_predictions.csv),
not from the subsequently refitted model bundle. Labels are joined from the
run's [labeled development rows](../results/codex-luna-verification-20261010-1/preparation/labeled_development.csv).
The prediction contract remains in [project_context.md](project_context.md):
churn means no orders during the following six months.

No model was fitted, no features were added to training, and no protected
October 2013 churn labels were constructed or evaluated. Raw order history is
restricted to the historical window before any diagnostic aggregation.

## 1. Which churners were missed?

Among 340 April accounts, there are 64 churners and 276 retained accounts.

| Result | Accounts | Median score | Median recency, days |
| --- | ---: | ---: | ---: |
| Detected churn | 8 | 0.9385 | 153 |
| Missed churn | 56 | 0.3122 | 17 |
| False alert | 14 | 0.9492 | 153 |
| Correctly retained | 262 | 0.3194 | 32 |

Recall is **12.5% (8/64)** and precision is **36.4% (8/22)**. There are 56
missed churners and 14 false alerts. These counts reconcile with the saved
selection manifest. The median score for all churners is 0.3183, versus 0.3471
for retained accounts: the overall distributions overlap substantially.
Scores are unverified probability estimates, not calibrated confidence levels.

The full missed-account list includes score, rank, threshold margin, history,
and diagnostic flags in [missed_churners.csv](../results/debugging/april_2013_precision_030/missed_churners.csv).
Rank 1 means the highest score among the 340 April accounts; ties share a rank.

### Distance from the threshold

These descriptive bands were specified before the detailed case comparisons.
They are not new alert policies or proposed business thresholds.

| Miss band | Definition | Misses | Median score |
| --- | --- | ---: | ---: |
| Near threshold | 0.67 <= score < 0.77 | 1 | 0.7422 |
| Low score | score < 0.30 | 23 | 0.1665 |
| Other miss | 0.30 <= score < 0.67 | 32 | 0.4676 |

Most misses are far below this operating point. A small threshold adjustment
would address few of them; any larger adjustment must also account for the
additional retained accounts alerted.

## 2. Recent activity is the principal failure group

"Recently active" means at least one order in [2013-01-01, 2013-04-01).
It is a historical subgroup, not a claim about the cause or type of churn.

| Historical activity | Accounts | Churners | Caught | Missed | False alerts | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Purchased in preceding 3 months | 314 | 54 | 0 | 54 | 0 | 0% |
| No purchase in preceding 3 months | 26 | 10 | 8 | 2 | 14 | 80% |

All 22 alerts are in the inactive group. That group contains 16 retained
accounts, of which 14 were alerted. High inactivity scores therefore catch
some churners while also producing substantial false alerts.

Among the 56 misses, 28 last ordered 2 days before the snapshot, 17 last ordered
32 days before, 9 last ordered 63 days before, and 2 last ordered 91 days before.
These are measured dates, not proof that recency alone caused the scores.

## 3. Did the purchase history contain warning signs?

Compare the latest quarter [2013-01-01, 2013-04-01) with the preceding quarter
[2012-10-01, 2013-01-01). A diagnostic spending decline is at least 25%, with
positive preceding-quarter revenue. If the preceding quarter has no revenue,
the ratio is undefined, not a decline. The 25% rule is an exploratory summary,
not a learned or validated feature-selection cutoff.

| Recently active group | Accounts | Revenue ratio defined | Decline >=25% | Flat/growing revenue | Undefined ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Missed churners | 54 | 44 | 9 | 33 | 10 |
| Retained accounts | 260 | 241 | 75 | 129 | 19 |

Two missed churners and 37 retained accounts have smaller declines. Among
accounts with defined ratios, large declines occur in 9/44 misses (20.5%) and
75/241 retained accounts (31.1%). **This particular decline flag is not enriched
among recently active misses.** It could still participate in interactions,
but the observed counts do not justify using it as a reliable churn rule.

Every recently active account in both groups placed exactly one order in the
latest quarter. None meets the 25% order-frequency decline flag. The median
available historical purchase gap is 91 days in both groups. Recent order count
and typical gap alone provide little separation within this April subgroup.

Sparse histories also limit the available explanation: 13 of the 56 misses
have at most two orders in the 12-month window; 4 have no estimable mean gap.
These are overlapping descriptive groups, not demonstrated causes of failure.

## 4. Representative cases and retained comparisons

For each miss, the script finds three retained April accounts closest in RFM
space. Recency stays in days; order count and revenue use log1p. Each coordinate
is scaled by the median and interquartile range from the fold's training
snapshots only, with a scale of 1 for a zero interquartile range. Euclidean
distance weights the three scaled coordinates equally. Ties use StoreID order.
Controls can be reused and can include false alerts. This is approximate
descriptive matching, not causal matching or an additional performance test.

These cases were selected to illustrate lowest score, near-threshold score,
large decline, and sparse history; they are not a random sample.

| Missed StoreID | Score | Latest-quarter revenue | Previous-quarter revenue | Closest retained StoreID | What the comparison shows |
| --- | ---: | ---: | ---: | ---: | --- |
| 872 | 0.6586 | 22.79 | 2,458.92 | 526 | Revenue falls 99.1%; the retained neighbor's revenue grows from 275.39 to 1,601.52. Their RFM scores are almost identical. A temporal spending feature describes an omitted difference here. |
| 304 | 0.6531 | 405.59 | 4,069.25 | 1324 | Revenue falls 90.0%; the retained neighbor also falls sharply, from 4,971.41 to 22.79. Decline alone would not distinguish this pair. |
| 1896 | 0.0489 | 97,597.12 | 88,446.80 | 816 | Four roughly quarterly orders, a purchase 2 days before the snapshot, and growing spending. The retained neighbor has similar cadence and spending. No clear warning in these aggregate descriptors. |
| 1112 | 0.7422 | 0 | 1,242.85 | 628 | The only near-threshold miss. It has two historical orders 184 days apart and recency 91 days. Its retained neighbor is a false alert with a shorter 92-day mean gap. |
| 336 | 0.1025 | 149.03 | 0 | 1936 | Only one order, 2 days before the snapshot. Cadence and revenue trend cannot be estimated; the retained neighbor also has one recent order. |

The monthly trajectories help make these interpretations reviewable:

| StoreID | Nonzero order months and revenue before April 2013 |
| --- | --- |
| 872 | Jul 2012: 20.52; Oct 2012: 2,458.92; Jan 2013: 22.79 |
| 304 | Jul 2012: 2,112.54; Oct 2012: 4,069.25; Jan 2013: 405.59 |
| 1896 | Jun 2012: 90,172.67; Sep 2012: 95,370.53; Dec 2012: 88,446.80; Mar 2013: 97,597.12 |
| 816, retained comparison | Jun 2012: 107,356.79; Sep 2012: 97,455.67; Dec 2012: 79,070.57; Mar 2013: 95,415.12 |
| 1112 | Jun 2012: 196.33; Dec 2012: 1,242.85 |
| 336 | Mar 2013: 149.03 |

All omitted months are zero-order months. Full histories, including zeros,
are saved in [monthly_purchase_history.csv](../results/debugging/april_2013_precision_030/monthly_purchase_history.csv).
The 56 misses have 168 comparisons in
[matched_retained_accounts.csv](../results/debugging/april_2013_precision_030/matched_retained_accounts.csv).
Nearest-control distance ranges from 0.00017 to 1.0170, with median 0.01785;
these are coordinate distances, not probabilities or guarantees of similarity.

## Interpretation and next decision

**Measured:** the selected RFM operating point misses every recently active
April churner. Some missed accounts show spending decline, many show flat or
growing spending, and similar retained accounts exhibit both patterns. Inactive
detected churners and false alerts have the same median 153-day recency.

**Supported interpretation:** this model and operating point do not separate
these groups well. Annual RFM totals omit some temporal differences, but a simple
recent spending-decline flag does not explain the whole failure group.

**Hypotheses to test later:** richer transaction descriptors or historically
available relationship/service signals may help separate recently active
churners from similar retained accounts. Sparse-history accounts may need
different contextual information. This report does not establish which feature,
architecture, or additional CompanyX table will supply that signal, nor that
the difficult accounts are inherently unpredictable.

The next decision is whether to compare existing richer feature sets under the
same recall/precision rule, or first investigate specific additional historical
fields for the recently active group. This diagnosis provides case IDs for that
investigation; no new training or database extraction is part of this step.

## Reproduction, outputs, and verification

From the repository root:

```bash
bi_venv/bin/python debug/diagnose_april_missed_churners.py
bi_venv/bin/python -m pytest -q debug/test_diagnose_april_missed_churners.py
```

The script writes only its generated artifacts under
`results/debugging/april_2013_precision_030/`. Rerunning replaces those diagnostic
artifacts. The source run, tracked datasets, and model files are read-only.

- [All 340 validation cases](../results/debugging/april_2013_precision_030/validation_cases.csv)
- [All 56 missed churners](../results/debugging/april_2013_precision_030/missed_churners.csv)
- [Group summaries](../results/debugging/april_2013_precision_030/cohort_summary.csv)
- [4,080 account-month history rows](../results/debugging/april_2013_precision_030/monthly_purchase_history.csv)
- [168 retained-account comparisons](../results/debugging/april_2013_precision_030/matched_retained_accounts.csv)
- [Diagnostic manifest and input/output hashes](../results/debugging/april_2013_precision_030/diagnostic_manifest.json)

Execution checks validate the development-input hash, complete unique joins,
label agreement, score bounds, recency and 3/6/12-month counts/revenue against
the source history, confusion counts against the saved selection manifest,
12 months per account, and 3 controls per miss. Focused tests cover score-band
boundaries, exclusion of orders at/after the snapshot and before the history
window, inclusion of zero-order months, and undefined ratios without an earlier
purchase. Matching checks cover deterministic ties and exclusion of validation
rows from distance normalization.

Verification completed: the real-data diagnostic executed successfully and all
four focused tests passed. Artifact hashes, report links, Python syntax, and
whitespace were checked. The full development pipeline and regression suite
were not run; the change is isolated to diagnostic code and documentation.

Limits: one selected model, one validation period, 64 churners, already inspected
development outcomes, approximate matching, and exploratory diagnostic cutoffs.
There is no causal conclusion, independently validated feature gain, database
availability assessment, or final-test performance claim in this report.

## Experiment: RFM versus seven features

Completed on 2026-10-10 after the diagnostic review. The agreed question was:
**does the existing seven-feature set catch more churners while maintaining at
least 30% precision in each of the same two development periods, particularly
among recently active accounts?**

**Result:** the seven-feature set improves mean fold recall from 28.25% to
35.03%, with precision above 30% in both folds. It catches a net four more
churners across the two periods, accompanied by six more false alerts. However,
it still misses all 54 recently active April churners. It improves this bounded
comparison but does not resolve the main April failure group.

### Controlled protocol

Only the feature bundle changes. Both models use weighted Logistic Regression,
C=1.0, max_iter=2000, and the original seed policy of seed 42. Median imputation,
standard scaling, class weights, and fitting use each fold's training rows only,
through the existing project helpers. The four fold-model fits are not refitted
on all development rows and no deployable model is selected by this script.

| Feature set | Inputs |
| --- | --- |
| RFM, 3 inputs | Recency, 12-month order count, 12-month revenue |
| Seven features, project name `pruned` | 3-month order count, 3-month revenue, recency, bike share, accessories share, clothing share, 12-month revenue |

The seven-feature bundle replaces the annual order count with recent order
count and adds recent revenue and product shares. This comparison cannot
attribute a gain to one individual feature.

| Fold | Training snapshots | Training rows / churners | Validation date | Validation rows / churners |
| --- | --- | --- | --- | --- |
| 1 | Jul 2012 | 326 / 76 | Jan 2013 | 343 / 25 |
| 2 | Jul and Oct 2012 | 692 / 123 | Apr 2013 | 340 / 64 |

For each feature set separately, the existing selector evaluates cutoffs 0.00
through 1.00 in steps of 0.01, using score >= cutoff. It chooses one cutoff used
in both folds to maximize mean recall, requiring precision >=0.30 in every fold.
Ties use worst-fold recall, mean precision, and stable candidate-threshold ID.
The selected cutoffs are 0.77 for RFM and 0.72 for the seven-feature set.
The floor and threshold grid were fixed before the experiment; no tuning,
additional feature creation, alternative architecture, or CompanyX extraction
was introduced.

The RFM scores reproduce the original saved predictions to a maximum absolute
difference of 1.11e-16, and its selected cutoff remains 0.77. This checks that
the comparison starts from the same baseline as the diagnosis.

### Overall metrics

| Period | Features | Churners caught / total | Recall | Precision | Missed churners | False alerts | Alerts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| January | RFM | 11 / 25 | 44.00% | 31.43% | 14 | 24 | 35 |
| January | Seven | 14 / 25 | 56.00% | 32.56% | 11 | 29 | 43 |
| April | RFM | 8 / 64 | 12.50% | 36.36% | 56 | 14 | 22 |
| April | Seven | 9 / 64 | 14.06% | 37.50% | 55 | 15 | 24 |

| Summary | RFM | Seven |
| --- | ---: | ---: |
| Mean fold recall, selection objective | 28.25% | 35.03% |
| Worst-fold recall | 12.50% | 14.06% |
| Mean fold precision | 33.90% | 35.03% |
| Pooled recall | 21.35% | 25.84% |
| Pooled precision | 33.33% | 34.33% |
| Pooled caught / missed churners | 19 / 70 | 23 / 66 |
| Pooled false alerts / all alerts | 38 / 57 | 44 / 67 |

Mean fold recall gives both periods equal weight. Pooled recall weights the
periods by their churn counts. Pooled figures count reseller-snapshot cases,
not necessarily distinct reseller identities. Both recall summaries improve,
but the larger gain is in January: +12 percentage points, compared with
+1.56 points in April. Precision improves modestly; false-alert workload still
increases because there are more alerts.

Average precision also improves: January 0.3014 to 0.3506 and April 0.2290 to
0.2494. These ranking metrics support an improvement beyond changing the
cutoff, but do not establish calibrated probabilities or independent model
quality. F1, accuracy, ROC-AUC, and alert rates are included in the generated
metrics table.

### Did it help the recently active group?

| Period | Historical group | Actual churners | Caught with RFM | Caught with seven | False alerts, RFM / seven |
| --- | --- | ---: | ---: | ---: | --- |
| January | Purchased in preceding 3 months | 13 | 1 | 3 | 5 / 14 |
| January | No purchase in preceding 3 months | 12 | 10 | 11 | 19 / 15 |
| April | Purchased in preceding 3 months | 54 | 0 | 0 | 0 / 0 |
| April | No purchase in preceding 3 months | 10 | 8 | 9 | 14 / 15 |

January recently active recall rises from 7.69% to 23.08%, with nine additional
false alerts in that subgroup. April recently active recall remains 0%; every
April alert from either model is still in the inactive group. The extra April
churner caught is StoreID 1112, the near-threshold inactive miss identified in
the original diagnosis.

The 30% constraint applies to each overall period, not to every subgroup.
January recently active precision is 16.67% for RFM and 17.65% for seven, below
30% in both. April recent-active precision has no empirical estimate because
neither model issues an alert there; the project metrics CSV represents this
zero-denominator case as 0 by convention.

### Which predictions changed?

| Change from RFM to seven | January cases | April cases |
| --- | ---: | ---: |
| Newly caught churners | 4 | 1 |
| Previously caught churners now missed | 1 | 0 |
| Added false alerts | 13 | 2 |
| Removed false alerts | 8 | 1 |
| Unchanged alert decisions | 317 | 336 |

January's net gain of three caught churners consists of newly caught StoreIDs
610, 1062, 1216, and 1466, offset by newly missed StoreID 912. Accounts 1062
and 1216 were recently active; 610 and 1466 were inactive. StoreID 1216's seven-
feature score is 0.7200318, barely above the fixed 0.72 cutoff, so this particular
decision should not be treated as robust to changes in fitted scores.

StoreID 912 illustrates a regression: its score falls from 0.9697 to approximately
2.83e-9. It has one historical order, no recent-quarter orders, annual revenue
91.176, and a clothing revenue share of 1.0. Those are descriptive facts;
this experiment does not isolate the cause of the score change or certify
that extreme score as a reliable probability.

In April, StoreID 1112 is newly caught. Retained accounts 524 and 1438 become
false alerts, while 1132's false alert is removed. All four have zero recent-
quarter orders. [Changed cases](../results/debugging/rfm_vs_seven_features_precision_030/changed_cases.csv)
retain both scores, both decisions, and recent activity for review.

### Conclusion and limits

Under the agreed fixed-model comparison, the seven-feature bundle is the better
development candidate: recall and precision improve in both periods, with no
period-level recall regression. The improvement is modest in April and does
not address its 54 recently active churners. Extra false alerts and the lost
January churner remain material tradeoffs.

This is evidence to retain the seven-feature result as a comparison baseline,
not to declare the debugging problem solved or replace a production model.
The two development periods have already guided diagnosis and threshold
selection. There are only 25 and 64 churners, repeated reseller identities,
one architecture, and fixed parameters. No significance claim, unseen-period
performance estimate, individual-feature benefit, or permanent churn cause is
established. Protected final-test outcomes were not accessed.

### Experiment reproduction and verification

```bash
bi_venv/bin/python debug/compare_rfm_seven_features.py
bi_venv/bin/python -m pytest -q debug tests/step_04_evaluation/test_recall_selection.py tests/step_04_evaluation/test_metrics_consistency.py
```

The experiment script writes its artifacts under
`results/debugging/rfm_vs_seven_features_precision_030/`; rerunning replaces only
those experiment artifacts. The original diagnostic outputs are preserved.

- [Experiment manifest: settings, selections, versions, hashes](../results/debugging/rfm_vs_seven_features_precision_030/experiment_manifest.json)
- [Per-period and subgroup metrics](../results/debugging/rfm_vs_seven_features_precision_030/fold_and_subgroup_metrics.csv)
- [Selection summaries](../results/debugging/rfm_vs_seven_features_precision_030/selection_summary.csv)
- [All 202 threshold evaluations](../results/debugging/rfm_vs_seven_features_precision_030/threshold_tradeoffs.csv)
- [Validation predictions for both feature sets](../results/debugging/rfm_vs_seven_features_precision_030/validation_predictions.csv)
- [All 683 paired cases](../results/debugging/rfm_vs_seven_features_precision_030/paired_cases.csv)
- [Changed cases](../results/debugging/rfm_vs_seven_features_precision_030/changed_cases.csv)
- [Prediction-change counts](../results/debugging/rfm_vs_seven_features_precision_030/prediction_change_counts.csv)
- [Matched training/evaluation counts](../results/debugging/rfm_vs_seven_features_precision_030/training_counts.csv)

Execution succeeded with four model fits and an RFM baseline reproduction check.
All **19 focused tests passed**, covering diagnostic boundaries, paired outcomes,
missing/mismatched populations and labels, no-feasible handling, subgroup
accounting, changed-baseline detection, and existing selection/metric helpers.
Output reconciliation, hashes, report links, syntax, and whitespace were checked.
The full development pipeline and complete regression suite were not run.

## CompanyX feature discovery and usefulness experiment — 2026-10-10

### Question and result

Can additional historical purchasing information improve recall beyond the
seven-feature baseline while keeping precision at least 30% in each development
period? We inventoried the local CompanyX restoration, extracted historical
order details, cleaned and reconciled them, constructed 11 new features, measured
their associations, and evaluated three fixed feature groups separately.

**None of the three groups improved mean recall in this experiment.** Discounts
kept the same 23 churners caught across the two periods and reduced false alerts
from 44 to 41. Basket features caught one previously missed active April churner,
but lost more churners elsewhere. These are development findings for the fixed
Logistic Regression setup, not evidence that every possible feature or model
using CompanyX data is ineffective.

### Database discovery and historical availability

Read-only SELECT queries used the existing local `companyx-audit` container and
`CompanyX_Audit` database. The inventory contains **72 tables, 491 columns, and
91 foreign-key column links**. The table count is not a count of relevant churn
sources; the saved column inventory supports further review.

| Source | Columns or information reviewed | Decision for this experiment |
| --- | --- | --- |
| `Sales.Customer` | `CustomerID`, `StoreID` | Join each reseller order to its established StoreID. |
| `Sales.SalesOrderHeader` | `SalesOrderID`, `OrderDate`, `SubTotal`, `SalesPersonID`, `TerritoryID` | Extract historical order facts and use the existing clean-order population. Salesperson and territory receive descriptive category rates only. |
| `Sales.SalesOrderDetail` | `SalesOrderDetailID`, `ProductID`, `OrderQty`, `UnitPrice`, `UnitPriceDiscount`, `LineTotal`, `SpecialOfferID` | Extract details; build quantity, basket, price, discount, and product-diversity features. Offer ID is retained for discovery, not modeled. |
| Header fulfillment fields | `Status`, `ShipDate`, `DueDate`, `ShipMethodID`, `OnlineOrderFlag` | Exclude from predictors: current status lacks a historical transition time; usable shipping duration and scheduled duration are constant. Ship method and online flag are also constant in this extract. |
| `Sales.Store` and current account profiles | `Demographics`, current assignments, `ModifiedDate` | Exclude changing profiles without historical versions. The earlier [source audit](data_analysis.md) records the common 2014 Store modification timestamp; this inventory confirms no temporal-table or CDC tracking for Store. |
| `Production.ProductReview` | Four rows, three products; no direct reseller key | No usable account-level complaint signal identified here. Product reviews are not a reseller service history. |
| Product price/cost histories and salesperson territory history | Dated history tables present in metadata | Catalogued but not extracted or modeled in this experiment. Historical realized order prices are already available in the extracted lines. Effective dates alone would still require checking recording-time availability before a later join. |
| Credit-card tables | Table/column metadata | No account payment-failure signal established from these tables; card credentials were not extracted. |

The inspected schema does not establish a linked history of support tickets,
service complaints, payment failures, or cancellation transitions. This is a
limitation of the reviewed sources, not proof that CompanyX has no such data in
other systems. All order facts are assumed to have been available when their
order was placed; a restored database does not verify this assumption through
recording-time history. Current mutable fields are not substitutes for that
history.

Source `Status` describes the **current** order state: 5 means shipped and 6 means
cancelled. There are 20 status-6 orders in the accepted history. We preserved the
existing eligibility and label contract, which does not filter by current status.
Changing that rule without a historical cancellation time could alter both
features and the meaning of churn. Whether these orders represent valid
purchases is an unresolved source/business-definition question for a separate
review; this experiment does not silently relabel accounts.

### Retrieval and cleaning

- Retrieved reseller orders strictly before **2013-04-01**: 1,892 headers and
  28,431 lines. No October final-test outcomes were retrieved or constructed.
- Retained **1,886 orders and 28,356 lines** using the established
  `datasets/orders_clean.csv` order IDs. Excluded IDs 43677–43682 remain excluded
  under the earlier cleaning contract, including the records with bad dates.
- Checked unique header/detail keys, reference uniqueness, no orphan lines,
  accepted orders having details, missing values, positive quantities/prices/net
  totals, and discounts in [0, 1]. Verified
  `LineTotal = OrderQty × UnitPrice × (1 − UnitPriceDiscount)`.
- Aggregated details by order before reconciling header revenue, quantity, line
  count, and average discount with the existing reference. Header `SubTotal`
  was never summed across duplicated detail joins. Revenue reconciliation allows
  0.02 currency units for stored rounding.
- Built **1,375 development account/snapshot rows**. Reconciled annual/recent
  order counts and annual revenue with the existing model inputs. After cleaning,
  every accepted order has a seven-day order-to-ship interval; this field cannot
  distinguish accounts in this extract.

### Feature definitions and missing values

Each window ends strictly before its prediction snapshot. A 3-month window is
the latest quarter; the previous quarter is months 3–6 before prediction; the
12-month window starts exactly 12 months before prediction. No label-window
purchases are included. The three groups were declared before examining their
correlations or fitting these candidates, and were tested separately.

| Group | Feature | Definition | Observed rows / 1,375 |
| --- | --- | --- | ---: |
| Basket volume | `avg_qty_3m` | Recent units / recent orders | 1,221 |
| Basket volume | `avg_qty_12m` | Annual units / annual orders | 1,375 |
| Basket volume | `avg_lines_3m` | Recent order lines / recent orders | 1,221 |
| Basket volume | `qty_ratio_3m` | Recent units / previous-quarter units | 1,096 |
| Basket volume | `net_unit_price_3m` | Recent net line revenue / recent units | 1,221 |
| Discounts | `weighted_discount_3m` | Sum(quantity × price × discount) / sum(quantity × price) in latest quarter | 1,221 |
| Discounts | `discount_change_3m` | Latest weighted discount minus previous-quarter weighted discount | 942 |
| Discounts | `discounted_line_share_12m` | Annual fraction of lines with a positive discount | 1,375 |
| Product diversity | `unique_products_3m` | Distinct products bought in latest quarter | 1,375 |
| Product diversity | `unique_products_12m` | Distinct products bought in preceding year | 1,375 |
| Product diversity | `new_product_share_3m` | Fraction of distinct recent products absent from the earlier nine months | 1,221 |

Distinct-product counts are zero when there are no purchases in the window.
Averages, prices, and discount rates are missing when their denominators do not
exist. Quantity ratios are missing without previous-quarter purchases; they are
zero when previous purchases exist but recent purchases do not. Discount change
requires purchases in both quarters. New-product share is missing without recent
purchases; “new” refers only to the observed earlier nine months, not the
account's lifetime. Training-only median imputation and missing-value indicators
handle these missing inputs.

### Association screening: what it found and what it means

Pearson correlation with the binary churn label (point-biserial correlation) and
Spearman rank correlation were calculated separately for each of four snapshots,
both overall and among accounts that bought in the preceding three months.
There are 88 feature/subgroup/period rows. Each row records observed and missing
counts, including the churners missing that feature. Constant features or
insufficient variation receive no correlation estimate.

Benjamini–Hochberg q-values adjust the finite p-values within each 11-feature,
period/subgroup/method family. This adjustment does not turn the many families,
the overlapping subgroups, or the subsequent model comparisons into one
confirmatory test. No pooled significance test treats repeated reseller
snapshots as independent. Previously inspected January/April labels are marked
as inspected validation periods. Correlations did not select the model features.

Examples from recently active accounts:

| Feature | October training Spearman rho | April validation Spearman rho | April within-family q |
| --- | ---: | ---: | ---: |
| Recent units per order | -0.285 | -0.121 | 0.075 |
| Annual units per order | -0.237 | -0.148 | 0.047 |
| Recent product diversity | -0.302 | -0.115 | 0.075 |
| Annual product diversity | -0.236 | -0.165 | 0.038 |
| Recent weighted discount | -0.125 | -0.047 | 0.425 |
| Recent quantity ratio | -0.104 | +0.105 | 0.123 |

Negative values mean higher feature values tended to accompany lower churn in
that period. They do not establish causality or predictive improvement over the
existing inputs. Several associations weaken or change direction by April.

Training-only redundancy screening also shows substantial overlap with existing
features: recent distinct-product count versus recent revenue has Spearman
rho **0.965**, recent units per order versus recent revenue **0.950**, and annual
product variety versus annual revenue **0.872**. This supports the explanation
that some candidates describe information the baseline already captures. It
does not prove redundancy caused the model results.

Salesperson/territory category churn rates are saved as descriptive tables,
not interpreted as numeric correlations between arbitrary IDs and labels.
Some categories are small; these fields were not used in the four model
configurations. Their incremental predictive value remains untested.

### Controlled predictive usefulness

Four configurations used the same weighted Logistic Regression (`C=1`,
`max_iter=2000`), established seed policy, training-only preprocessing, labels,
and temporal folds: July 2012 training to January 2013 validation; July plus
October 2012 training to April 2013 validation. No parameter tuning or combined
three-group search was performed. There are eight fits per complete run.

For each configuration, one threshold was selected over 0.00–1.00 in 0.01 steps
using the existing rule: maximize mean fold recall while requiring precision
at least 30% in **both overall validation periods**. This constraint does not
apply separately to the recently active subgroup. The original seven-feature
baseline was reproduced, including scores within 1e-10, labels, matched cases,
and the selected 0.72 threshold.

| Features | Threshold | January recall / precision | April recall / precision | Mean recall | Churners caught, summed | False alerts, summed |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Seven-feature baseline | 0.72 | 56.00% / 32.56% | 14.06% / 37.50% | **35.03%** | 23 | 44 |
| Baseline + basket volume | 0.85 | 40.00% / 35.71% | 12.50% / 30.77% | 26.25% | 18 | 36 |
| Baseline + discounts | 0.74 | 56.00% / 35.00% | 14.06% / 37.50% | **35.03%** | 23 | 41 |
| Baseline + product diversity | 0.81 | 44.00% / 35.48% | 14.06% / 37.50% | 29.03% | 20 | 35 |

Mean recall is the equal-weight average of the two period recalls, not recall
calculated from pooled counts. Summed counts represent account/snapshot cases,
not necessarily distinct resellers. F1, accuracy, ROC-AUC, average precision,
and alert rates are in the metrics artifact.

April average precision increases slightly from 0.2494 to 0.2625 with basket
features, 0.2527 with discounts, and 0.2531 with diversity. This does not translate
into better recall at the constrained operating point. For example, basket
features improve this April ranking metric but lower January recall and require
a higher common threshold to meet the precision rule.

### What happened to the hard-to-predict churners?

| Features | January active churners caught / 13 | April active churners caught / 54 | April active false alerts |
| --- | ---: | ---: | ---: |
| Baseline | 3 | 0 | 0 |
| + Basket volume | 0 | 1 | 3 |
| + Discounts | 3 | 0 | 0 |
| + Product diversity | 0 | 0 | 0 |

Basket features newly catch April StoreID **902**, a recently active churner.
However, they lose April inactive churners **418 and 1270**, and four January
churners (**916, 1062, 1216, 1364**). The net loss is five caught churners across
the two periods. Its April active precision is 25% (one true alert among four),
below the overall-period floor; its active recall is only 1.85%.

Discount features catch exactly the same churners as the baseline. In January
they remove six false alerts and introduce three, for a net reduction of three.
Every April alert decision remains unchanged. Product diversity loses three
January active churners; its April alert decisions also remain unchanged.

### Conclusion and next decision

The purchasing features tested here do **not** resolve the recall problem.
Discounts offer a small measured false-alert reduction without losing recall,
but no improvement in catching the 54 April active churners. Basket and diversity
groups should not replace the baseline under the current primary metric.

These results apply to the fixed linear model, these feature definitions, and
the inspected development periods. They do not establish calibrated scores,
independent generalization, individual-feature effects, or usefulness under
nonlinear interactions. Threshold selection and the three comparisons reuse
development labels; the differences are exploratory, not a significance claim.
Historical order-time availability remains an assumption. Protected final-test
labels were not accessed, and the methodology is not frozen for final evaluation.

The next investigation should resolve purchase/cancellation semantics and seek
historically timestamped account signals outside these purchasing summaries
(for example service or payment issues, if another source actually contains
them). A separately planned nonlinear comparison could test interactions in
the current features. Neither step has been executed here; further correlations
alone would not establish incremental recall.

### Artifacts, reproduction, and verification

All new scripts and focused tests are in root `debug/`. Generated artifacts are
under `results/debugging/companyx_features/`, and existing experiments are
preserved. Inventory/extraction require access to the existing local Docker
restoration and use its configured credential without printing it.

```bash
bi_venv/bin/python debug/inventory_companyx.py
bi_venv/bin/python debug/extract_companyx_history.py
bi_venv/bin/python debug/build_companyx_features.py
bi_venv/bin/python debug/evaluate_companyx_features.py
bi_venv/bin/python debug/verify_companyx_outputs.py
bi_venv/bin/python -m pytest -q debug tests/step_04_evaluation/test_recall_selection.py tests/step_04_evaluation/test_metrics_consistency.py
```

- [Table inventory](../results/debugging/companyx_features/source/tables.csv),
  [column inventory](../results/debugging/companyx_features/source/columns.csv),
  [foreign keys](../results/debugging/companyx_features/source/foreign_keys.csv),
  and [order-column descriptions](../results/debugging/companyx_features/source/column_descriptions.csv).
- [Source profiles](../results/debugging/companyx_features/source/source_profiles.csv)
  and [cleaning manifest](../results/debugging/companyx_features/cleaning_manifest.json).
- [Declared feature groups](../results/debugging/companyx_features/feature_protocol.json)
  and [development features](../results/debugging/companyx_features/development_features.csv).
- [Numeric associations and adjusted p-values](../results/debugging/companyx_features/numeric_associations.csv),
  [category rates](../results/debugging/companyx_features/categorical_churn_rates.csv),
  and [training redundancy](../results/debugging/companyx_features/training_feature_redundancy.csv).
- [Experiment settings and reproducibility hashes](../results/debugging/companyx_features/experiment_manifest.json),
  [selection summary](../results/debugging/companyx_features/selection_summary.csv),
  [period/subgroup metrics](../results/debugging/companyx_features/fold_and_subgroup_metrics.csv),
  and [all 404 threshold evaluations](../results/debugging/companyx_features/threshold_tradeoffs.csv).
- [Validation predictions](../results/debugging/companyx_features/validation_predictions.csv),
  [paired cases](../results/debugging/companyx_features/paired_cases.csv),
  [changed decisions](../results/debugging/companyx_features/changed_cases.csv),
  and [change counts](../results/debugging/companyx_features/prediction_change_counts.csv).

Extraction, cleaning, feature construction, and model evaluation completed.
**31 focused tests passed**, including temporal endpoints, missing denominators,
protected snapshot rejection, duplicate/missing detail checks, correlation
edge cases, p-value adjustment, matched labels/populations, and existing
selection/metric tests. Confusion counts, subgroup partitions, paired changes,
hashes, report links, Python syntax, and whitespace were checked. The full
development pipeline, complete regression suite, and protected final evaluation
were not run.

## Seven versus ten features across models and validation methods — 2026-10-10

### Purpose and executed protocol

The question is whether the three discount features add predictive value to
the seven-feature baseline, across models and across two validation methods.
This comparison uses the previously cleaned CompanyX development features.
No additional source extraction, feature selection, resampling, or parameter
tuning was performed.

The seven inputs are recent order count, recent revenue, recency, bike share,
accessory share, clothing share, and annual revenue. The ten-input version adds
`weighted_discount_3m`, `discount_change_3m`, and
`discounted_line_share_12m`, with the definitions and historical-availability
assumption recorded in the preceding CompanyX experiment. Missing-value
indicators added by preprocessing are additional transformed inputs, not new
business features.

| Architecture | Fixed configuration | Preprocessing |
| --- | --- | --- |
| Logistic Regression | C=1, max_iter=2000; one deterministic fit per feature set/fold | Training-only median imputation with indicators, then standard scaling |
| Random Forest | 300 trees, unlimited depth, min_samples_leaf=1, max_features=sqrt | Training-only median imputation with indicators |
| Histogram Gradient Boosting | learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0, early_stopping=False | Existing native missing-value handling |

Every architecture uses balanced sample weights computed from its own training
labels. The tree models use seeds **42, 78, 88, 1034, 2026**; their primary
predictions average the five probability estimates. Logistic Regression fits
once, matching the established deterministic baseline. Individual tree-seed
results are also evaluated, without selecting a preferred seed.

The full run completed **154 fits**: 44 temporal fits plus 110 grouped fits.
Thread counts were bounded with `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`; model parameters and Random Forest's existing
parallelism policy were retained. The protocol and shared fold assignments
were saved before fitting. The prior temporal Logistic Regression scores for
both feature sets reproduce with maximum absolute error **1.11e-16**, with
matched populations and labels.

### Two validation questions

**Temporal validation is the primary comparison for future churn.** It uses the
same approved folds and checks that training labels are available by prediction
time. Resellers may recur across periods, as appropriate for forecasting
existing accounts.

| Temporal fold | Training rows / churn cases | Validation rows / churn cases |
| --- | --- | --- |
| July 2012 to January 2013 | 326 / 76 | 343 / 25 |
| July + October 2012 to April 2013 | 692 / 123 | 340 / 64 |

**Grouped k-fold evaluates held-out resellers across mixed periods.** We use
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with StoreID as
the group. Every reseller's development snapshots stay together, no reseller
appears on both sides of a fold, and every development row is validated once.
Stratification balances row-level labels approximately; it does not create a
single lifetime churn label for each reseller.

| Grouped fold | Validation rows | Validation resellers | Churn cases | Churn rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 272 | 86 | 42 | 15.44% |
| 2 | 288 | 87 | 43 | 14.93% |
| 3 | 285 | 86 | 42 | 14.74% |
| 4 | 270 | 80 | 43 | 15.93% |
| 5 | 260 | 83 | 42 | 16.15% |

Grouped validation includes all **1,375 rows, 422 resellers, and 212 churn
cases** from the four approved snapshots. Temporal validation includes **683
January/April rows and 89 churn cases**. Grouped training also has more rows per
fold and may include later-period records than a held-out record. These methods
therefore differ in targets, training size, chronology, and reseller separation.
Their scores are not a matched-target experiment proving one method or model
has better future performance. Grouped evaluation does not enforce historical
training-label availability; its interpretation is retrospective generalization
to held-out reseller identities.

### Thresholds and reporting rules

For each architecture, feature set, and validation method, one common threshold
is selected over 0.00–1.00 in 0.01 steps. The unchanged project rule maximizes
mean fold recall while requiring **precision at least 30% in every overall
fold**, with ties broken by worst-fold recall, mean precision, and the stable
configuration/threshold identifier. This floor does not apply to every subgroup.
Infeasible configurations would be reported rather than given a fallback
threshold. All 12 primary configurations were feasible.

Individual-seed evaluations select thresholds using the same rule and are
reported separately from the averaged predictions. There are 12 primary and
44 individual-seed operating summaries, and 5,656 threshold-grid rows. Mean
recall/precision are equal-weight fold averages; summed caught/false-alert counts
refer to account/snapshot cases, not necessarily distinct resellers.

### Temporal results: primary evidence

| Model | Features | Threshold | January recall / precision | April recall / precision | Mean recall | Caught / false alerts, summed |
| --- | ---: | ---: | --- | --- | ---: | --- |
| Logistic Regression | 7 | 0.72 | 56.00% / 32.56% | 14.06% / 37.50% | 35.03% | 23 / 44 |
| Logistic Regression | 10 | 0.74 | 56.00% / 35.00% | 14.06% / 37.50% | 35.03% | 23 / 41 |
| Random Forest | 7 | 0.59 | 56.00% / 30.43% | 17.19% / 35.48% | **36.59%** | 25 / 52 |
| Random Forest | 10 | 0.71 | 36.00% / 33.33% | 10.94% / 30.43% | 23.47% | 16 / 34 |
| Histogram Gradient Boosting | 7 | 0.93 | 20.00% / 55.56% | 1.56% / 33.33% | 10.78% | 6 / 6 |
| Histogram Gradient Boosting | 10 | 0.58 | 52.00% / 32.50% | 15.63% / 31.25% | 33.81% | 23 / 49 |

- **Logistic Regression:** unchanged recall; three fewer false alerts. Exactly
  the same churners are caught. Its prior small precision benefit reproduces.
- **Random Forest:** ten features lose nine previously caught churners and
  catch no new ones at their selected thresholds. Removing 19 false alerts and
  adding one saves 18 alerts but sacrifices recall in both periods.
- **Gradient boosting:** ten features newly catch 17 churners and retain all six
  previously caught ones, while adding 43 false alerts. Recall improves by
  23.03 percentage points on average and in both periods. This benefit is
  specific to the constrained operating point: the common threshold falls from
  0.93 to 0.58, while mean average precision changes only from 0.3091 to 0.3124.
  The large threshold-dependent recall improvement is not a comparable jump in
  ranking quality or evidence of calibrated probabilities.

Seven-feature Random Forest has the highest mean temporal recall in this
bounded comparison. Compared with ten-feature Logistic Regression, it catches
two more churn cases but produces 11 more false alerts. Both satisfy the overall
precision floor. This is a development candidate comparison, not a production
replacement or independent confirmation.

### Grouped k-fold results: supporting evidence

| Model | Features | Threshold | Mean recall | Mean precision | Caught / false alerts, summed |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 7 | 0.49 | 54.31% | 38.26% | 115 / 189 |
| Logistic Regression | 10 | 0.49 | 56.18% | 38.74% | 119 / 193 |
| Random Forest | 7 | 0.26 | **75.55%** | 34.47% | 160 / 318 |
| Random Forest | 10 | 0.26 | 74.56% | 34.50% | 158 / 310 |
| Histogram Gradient Boosting | 7 | 0.13 | 70.86% | 37.39% | 150 / 257 |
| Histogram Gradient Boosting | 10 | 0.06 | 75.51% | 33.45% | 160 / 324 |

Adding discounts raises Logistic Regression mean recall by 1.87 points, with
13 newly caught and nine newly missed cases; it adds four false alerts net.
Gradient boosting gains 4.65 points, with 14 newly caught and four newly missed
cases, at the cost of 67 additional false alerts. Its fold-1 and fold-5 recalls
decrease despite the mean increase. Logistic Regression also loses recall in
fold 3. Random Forest mean recall falls by 0.99 points, with four newly caught
and six newly missed cases, and eight fewer false alerts.

The higher grouped scores do not fix the weaker future-period results. They
apply to a different evaluation question. Also, mean average precision falls
when adding discounts for all three grouped models (Logistic Regression
0.5133 to 0.5074, Random Forest 0.4722 to 0.4631, gradient boosting 0.5034 to
0.4827). Higher constrained recall can coexist with weaker average precision
and a larger alert workload. Full per-fold metrics include F1, accuracy,
ROC-AUC, average precision, and alert rates.

### Recently active churners and seed sensitivity

| Model | Features | January active caught / 13 | April active caught / 54 | April active false alerts |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 7 | 3 | 0 | 0 |
| Logistic Regression | 10 | 3 | 0 | 0 |
| Random Forest | 7 | 4 | **6** | 5 |
| Random Forest | 10 | 2 | 3 | 4 |
| Histogram Gradient Boosting | 7 | 1 | 0 | 0 |
| Histogram Gradient Boosting | 10 | 4 | 3 | 7 |

The seven-feature forest catches six of the 54 hard April churners (11.11%
active recall), with five active false alerts. Ten-feature boosting catches
three (5.56%), with seven false alerts. None resolves the active-churner problem:
even the best configuration on this subgroup still misses **48 of 54**.

In grouped evaluation, among 121 active churn cases, Logistic Regression catches
24 to 29 when adding discounts; the forest catches 71 to 68; boosting catches
63 to 71. These counts include earlier periods and cannot be interpreted as
April future-prediction improvements.

Random Forest's discount benefit varies across seeds: it improves mean recall
for two of five seeds in temporal validation and three of five in grouped
validation, but its averaged-prediction result declines under both methods.
Temporal single-seed recall differences range from -12.34 to +5.91 percentage
points; grouped differences range from -2.39 to +1.84 points. This is evidence
against claiming a stable forest benefit from these features.

Boosting predictions are identical across the five seeds in this setup, so its
repeated seed results do not provide independent evidence of robustness.
Logistic Regression is fitted once per fold. Seed checks do not assess
sensitivity to a different grouped fold assignment or an unseen time period.

### Decision and limits

**Do not adopt ten features universally across architectures.** Retain discounts
as a candidate for Logistic Regression's false-alert reduction and for the
boosting model's measured recall improvement. Keep the seven-feature forest as
the leading temporal recall candidate in this comparison, with its larger alert
workload and seed variability disclosed. A global feature-set change or model
promotion has not been made.

These are fixed-parameter, balanced-weight experiments on development labels
already inspected during prior debugging. Thresholds are selected and described
on the same validation predictions; the resulting scores are exploratory and
may be optimistic. Model and feature comparisons reuse those labels, and no
significance claim or unseen-period performance estimate is established.
Correlation screening did not choose these configurations. The original
historical order-time availability assumption remains unverified by version
history. Protected October final-test labels were not accessed.

### Artifacts and verification

Scripts and focused tests are in root `debug/`; all new generated outputs are
in `results/debugging/seven_vs_ten_features/`. Existing CompanyX and earlier
diagnostic outputs are preserved. Reproduction uses the saved cleaned features:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/compare_seven_ten_features.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python -m pytest -q debug tests/step_04_evaluation/test_recall_selection.py tests/step_04_evaluation/test_metrics_consistency.py
bi_venv/bin/python debug/verify_seven_ten_outputs.py
```

- [Frozen protocol](../results/debugging/seven_vs_ten_features/experiment_protocol.json)
  and [execution manifest, versions, and hashes](../results/debugging/seven_vs_ten_features/experiment_manifest.json).
- [Shared fold membership](../results/debugging/seven_vs_ten_features/fold_membership.csv)
  and [training/validation counts](../results/debugging/seven_vs_ten_features/fold_counts.csv).
- [Operating selections](../results/debugging/seven_vs_ten_features/selection_summary.csv),
  [fold/subgroup metrics](../results/debugging/seven_vs_ten_features/fold_and_subgroup_metrics.csv),
  and [all threshold evaluations](../results/debugging/seven_vs_ten_features/threshold_tradeoffs.csv).
- [Primary feature differences](../results/debugging/seven_vs_ten_features/ensemble_feature_deltas.csv)
  and [individual-seed differences](../results/debugging/seven_vs_ten_features/seed_feature_deltas.csv).
- [Averaged predictions](../results/debugging/seven_vs_ten_features/validation_predictions.csv)
  and [individual-seed predictions](../results/debugging/seven_vs_ten_features/individual_seed_predictions.csv).
- [Paired cases](../results/debugging/seven_vs_ten_features/paired_cases.csv),
  [changed cases](../results/debugging/seven_vs_ten_features/changed_cases.csv),
  and [prediction-change counts](../results/debugging/seven_vs_ten_features/prediction_change_counts.csv).

All **39 focused tests passed**, including model smoke fits with missing
discounts, training-only preprocessing, reproducible reseller-disjoint folds,
chronological label availability, infeasible threshold handling, matched labels,
and existing feature/diagnostic/selection tests. Independent saved-output
verification reconciled all 154 fits, source/code/output hashes, fold coverage,
seed averaging, confusion counts, precision constraints, recall summaries,
paired changes, report links, Python syntax, and whitespace. The previous
CompanyX output verification also still passes. The complete regression suite,
full development pipeline, parameter tuning, and protected final evaluation
were not run.

## Final testing evaluation — 2026-10-11

### Authorization and frozen methodology

The user explicitly requested final testing and chose **grouped k-fold's 30%
precision rule as primary**, plus the existing 20%, 40%, and 50% rules as
secondary results. This satisfies the final-evaluation authorization and
methodology-freeze requirements. The 60% scenario was infeasible in development
and was not evaluated with an invented test threshold.

Before accessing outcomes, the protocol froze ten-feature histogram gradient
boosting, the existing baseline parameters (learning_rate=0.1, max_iter=100,
max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0, early_stopping=False),
native missing handling, balanced training sample weights, and the five-seed
probability average. Training uses all **1,375 approved development rows**
(July/October 2012 and January/April 2013, with 212 churn cases). Every training
label window resolves by the final prediction date.

The fixed cutoffs were selected only from saved grouped development predictions:
20% → **0.01**, primary 30% → **0.06**, 40% → **0.55**, 50% → **0.88**. They were
not changed, searched, or selected using test outcomes. Development precision
floors are selection constraints, not guarantees about test precision.

### Prediction checkpoint and source reconciliation

Preparation extracted only reseller orders before **2013-10-01**. It retained
2,723 clean history orders and 44,501 detail lines, excluding the same six
established invalid orders. The seven base features reproduced the tracked
492-row unlabeled final reference. The three discounts used only historical
order lines; recent weighted discount is missing for 45 accounts and discount
change for 159. The existing order-time availability and cancellation/purchase
contract assumptions remain unchanged.

Five models were refitted and saved under
`models/provisional/gb10_grouped_20261011/ensemble.joblib`. Reloaded predictions
match the original scores. **All 492 unlabeled scores and all four alert
decisions were saved and hashed before outcome access.** The second stage checks
the protocol, input, code, model, and prediction hashes before revealing labels;
it fits no models.

The label window is **[2013-10-01, 2014-04-01)**. Its 896 reseller orders were
reconciled directly against the CompanyX restoration and tracked clean CSV,
including IDs, account mapping, dates, subtotals, quantities, and line counts.
The standard eligible-account/half-open label contract produced **492 test
resellers, 46 churners, and 446 retained accounts**, a churn prevalence of 9.35%.

The first preparation attempt encountered the stopped local audit container
before any outcomes were accessed. Its frozen protocol was preserved in
`results/final_evaluation/gb10_grouped_20261011_preparation_attempt_1/`. Starting
the existing container allowed the successful fresh preparation. This was not
a failed or repeated test-label evaluation.

### Final results

These metrics describe one final cohort, not averages across validation folds.
Minimum precision is the **development rule used to freeze the cutoff**.

| Rule | Minimum development precision | Test recall | Actual test precision | F1 | Total churners | Churners caught | False alarms | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Secondary | 20% | 97.83% | 20.64% | 0.341 | 46 | 45 | 173 | 1 |
| **Primary** | **30%** | **93.48%** | **28.48%** | **0.437** | **46** | **43** | **108** | **3** |
| Secondary | 40% | 71.74% | 56.90% | 0.635 | 46 | 33 | 25 | 13 |
| Secondary | 50% | 43.48% | 76.92% | 0.556 | 46 | 20 | 6 | 26 |

Churners caught are true positives; false alarms are retained accounts receiving
alerts. The primary rule has
151 alerts, 338 true negatives, and 77.44% accuracy. Its recall is
`43 / 46 = 93.48%`, while precision is `43 / 151 = 28.48%`.
**The primary rule fails the intended 30% precision requirement on this test.**
It is about 1.52 percentage points below that level. The predefined secondary
rules are reported as secondary evidence; no new operating winner is selected
from their test results.

### Recently active accounts and interpretation

At the primary cutoff, 447 accounts purchased in the preceding three months:
11 churned, of which eight were caught and three missed. There were **98 active
false alerts**, giving active recall **72.73%** but precision only **7.55%**.
The remaining 45 inactive accounts contain 35 churners; all 35 are caught with
ten false alerts, giving 100% recall and 77.78% precision.

Thus overall recall does not imply good precision for recently active accounts.
This test also differs markedly from April development: 35/46 test churners
are inactive, compared with 10/64 April churners. That observed population mix
can help explain why October is easier at the chosen rule; this is a supported
explanation, not an isolated causal analysis. Test prevalence and model
refitting also differ from grouped validation. The large increase from grouped
mean recall 75.51% to test recall 93.48% is not evidence of stable performance
across future periods or calibrated individual risk estimates.

**October 2013 outcomes are now inspected.** Any subsequent changes informed by
this report must disclose test reuse and obtain a new untouched period for an
independent final performance claim. The standard development guards still
exclude these labels. The frozen model remains provisional and has not been
deployed.

### Recommended follow-up — not yet executed

The next diagnosis should prioritize the **98 false alarms among recently
active accounts** and the **three missed active churners**. Compare their
historical purchasing and discount patterns with correctly classified active
accounts to identify candidate explanations. These comparisons are exploratory
because the final outcomes have now been inspected.

Use those hypotheses to design controlled development experiments, measuring
recall, actual precision, F1, churners caught, and false alarms overall and by
recent activity. Retain both temporal and grouped validation; temporal results
remain the evidence for predicting a later period. Any feature or threshold
changes must be selected using development data, and a new untouched cohort
must be reserved for independent final evaluation. The secondary test results
alone do not establish which rule should be deployed.

### Artifacts and verification

The entry point is [run_final_testing_evaluation.py](../debug/run_final_testing_evaluation.py).
Its separate `--stage prepare` and `--stage evaluate` commands enforce the
prediction-before-label checkpoint; both preserve the fixed run and reject
overwriting an existing preparation or completed evaluation. The historical
commands have already completed and should not be rerun to create a fresh test.
Use saved-output verification instead:

```bash
bi_venv/bin/python debug/verify_final_testing_outputs.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python -m pytest -q debug tests/step_04_evaluation/test_recall_selection.py tests/step_04_evaluation/test_metrics_consistency.py
```

- [Frozen protocol and authorization](../results/final_evaluation/gb10_grouped_20261011/frozen_protocol.json).
- [Pre-label prediction checkpoint](../results/final_evaluation/gb10_grouped_20261011/prediction_checkpoint.json)
  and [unlabeled predictions](../results/final_evaluation/gb10_grouped_20261011/unlabeled_predictions.csv).
- [Final metrics](../results/final_evaluation/gb10_grouped_20261011/final_test_metrics.csv)
  and [labeled predictions](../results/final_evaluation/gb10_grouped_20261011/final_test_predictions.csv).
- [Primary error cases](../results/final_evaluation/gb10_grouped_20261011/primary_error_analysis.csv)
  and [execution/source reconciliation manifest](../results/final_evaluation/gb10_grouped_20261011/evaluation_manifest.json).

**47 focused tests passed**, including eight synthetic final-stage safeguards.
The synthetic tests do not inspect final labels. Independent artifact checks
verified frozen-code/data hashes, five-model seed averaging, prediction-before-
outcome chronology, all 12 overall/subgroup metric rows, unchanged rule roles
and cutoffs, confusion counts, average precision, ROC-AUC, and output hashes.
Final preparation and source-reconciled evaluation completed successfully.
The full regression suite, supported full pipeline, parameter tuning, and any
post-test model/threshold changes were not run.

## SubTotal versus TotalDue: planned grouped comparison — 2026-10-14

Status: protocol recorded before implementation or training; matched data
preparation and the grouped comparison are complete. Work remains on
`experimental_research`;
the fixed pipeline on `main` is unchanged.

### Question and controlled change

Does replacing merchandise spending with the full order amount improve
development recall for seven-feature Random Forest or ten-feature histogram
gradient boosting under the existing 30% per-fold precision constraint?

Compare four configurations: each architecture with SubTotal and with TotalDue.
The forest uses `PRUNED_FEATURES`; boosting adds the three discount features
defined in `debug/build_companyx_features.py`. Replace only `revenue_3m` and
`revenue_12m` with sums of TotalDue in the alternative. Preserve column order,
all other values, eligibility, labels, and account/snapshot populations.
Category-share denominators remain SubTotal, and discount formulas remain
unchanged. Do not add tax/freight ratios or tune model settings in this experiment.
Interpret revenue-definition differences within each architecture; differences
between architectures also include their different feature sets.

### Sources and preparation safeguards

Use the saved `results/debugging/companyx_features/development_features.csv`
as the baseline, checking its recorded input hash before use. Use accepted IDs
from `datasets/orders_clean.csv` to define the order population. Read CompanyX
order headers through the existing read-only helper, restricted to orders
strictly before 2013-04-01, the latest feature snapshot in this experiment.
Extract order ID, reseller ID, order date, SubTotal, TaxAmt, Freight, and TotalDue.
Do not extract October test outcomes or build new final-test labels.

Require unique header IDs, complete matching accepted historical orders, matching
reseller/date/subtotal values, and finite positive accepted amounts. Check
`TotalDue = SubTotal + TaxAmt + Freight` within rounding tolerance. Save the
extracted historical headers and their hashes under a new experiment directory.
Aggregate only orders in each snapshot's half-open three- and twelve-month
history windows; no recent orders means zero recent revenue. Verify exact parity
for labels, keys, and every unchanged feature, including missing discount values.
Retain the documented assumption that order-time monetary facts were available
at prediction time; a later database extract does not prove recording chronology.

### Training and validation

At the user's request, use grouped validation only; temporal validation is
outside this experiment. Match the existing seven-versus-ten grouped setup:
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by
`StoreID`. This is stratified grouped validation, not plain `GroupKFold`.

Use all 1,375 development rows from July/October 2012 and January/April 2013,
covering 422 reseller identities and 212 churn cases. Sort by snapshot and
StoreID before generating the splits, matching the previous implementation.
Generate folds once and use identical memberships for all four configurations.
Keep every snapshot of a reseller together; require disjoint training/validation
reseller identities, both classes on each side, and exactly one held-out
prediction per development row per configuration.

These folds evaluate unseen reseller identities across mixed historical periods.
They do not enforce training-label availability by each held-out snapshot and
must not be interpreted as later-period performance estimates. Historical
feature windows still exclude information at or after their own snapshot.
Reuse `ModelFactory` and the baseline settings used in the seven-versus-ten
comparison:

- Forest: 300 trees, unlimited depth, minimum leaf size 1, `max_features="sqrt"`,
  training-only median imputation with missing indicators.
- Boosting: learning rate 0.1, 100 iterations, 31 maximum leaf nodes, minimum
  leaf size 20, L2 regularization 0, early stopping disabled, native missing values.
- Both: balanced sample weights computed from each training fold's labels;
  seeds 42, 78, 88, 1034, and 2026; average predicted probabilities across seeds.

This requires 100 fits: two architectures × two definitions × five folds × five
seeds. No tuning, resampling, additional feature search, or final refitting.

### Selection and reporting

For each of the four ensemble configurations, independently choose one common
cutoff across all five folds using `evaluation/selection.py`: search 0.00–1.00 in
0.01 steps, maximize mean fold recall with precision at least 0.30 in every fold,
and break ties by worst-fold recall, mean precision, then stable selection ID.
Report infeasible configurations rather than relaxing the floor.

Save fold memberships, individual-seed and ensemble probabilities, all cutoff
tradeoffs, selected cutoffs, and matched account-level decision changes. Report
per-fold and pooled confusion counts, precision, recall, F1, alert counts/rates,
average precision, and ROC-AUC; distinguish fold means from pooled scores.
Apply each ensemble's selected cutoff to recent-active and inactive subgroups
using the existing three-month order-count definition. Report seed variation
at the corresponding ensemble cutoff without choosing a preferred seed.
Also compare both definitions at the SubTotal ensemble's selected cutoff as a
secondary diagnostic, separating score changes from cutoff-selection changes.

Use a new `results/debugging/subtotal_vs_totaldue/<run_id>/` directory and reject
existing output directories. Record protocol, input/code hashes, parameters,
runtime versions, fitted-model count, and checks. Preserve earlier experiments.
Verify fold memberships and baseline consistency against saved seven-versus-ten
grouped predictions;
investigate differences rather than assuming historical numerical parity.

Compare recall changes alongside false alerts, fold/subgroup variation, and
ranking quality. These are exploratory comparisons on previously inspected
development labels, not significance claims or independent test estimates.
Do not promote a replacement into `main` from this experiment alone.

### Completed matched data preparation — 2026-10-14

`debug/prepare_subtotal_totaldue.py` implements the source reconciliation and
paired revenue construction above. Run it with a new ID:

```bash
bi_venv/bin/python debug/prepare_subtotal_totaldue.py --run-id <new-run-id>
bi_venv/bin/python -m pytest -q debug/test_prepare_subtotal_totaldue.py
```

The completed run is under
`results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01/`.
It contains the raw historical headers, accepted headers, SubTotal and TotalDue
development CSVs, a copy of the protocol, and a hashed preparation manifest.
CompanyX returned 1,892 reseller orders before 2013-04-01; 1,886 are accepted,
and the six established excluded orders remain excluded. Both datasets have
1,375 rows, 422 reseller identities, and 212 churn cases. TotalDue changes
three-month revenue for 1,221 rows and twelve-month revenue for all 1,375 rows.

All 13 focused tests passed, covering half-open history boundaries, zero recent
revenue, unchanged missing discount values, source mismatches, missing/duplicate
orders, invalid amounts, and overwrite protection. The real-data preparation
passed baseline hash/count checks, source reconciliation, and SubTotal
reconstruction. A separate saved-artifact check confirmed exact full baseline
parity, unchanged-column parity between variants, and every output hash.
No models were fitted during preparation and no October test outcomes were
accessed. The subsequent grouped comparison is recorded below.

### Completed grouped revenue comparison — 2026-10-14

The runner `debug/compare_subtotal_totaldue.py` completed all 100 planned fits
under `results/debugging/subtotal_vs_totaldue/revenue-comparison-20261014-01/`.
Its shared fold memberships match the earlier seven-versus-ten grouped run
exactly. Reproduced SubTotal ensemble scores differ from saved scores by at
most 3.34e-16 for forest and 1.12e-16 for boosting. All four configurations
meet the 30% precision floor in every fold at their selected cutoffs.

| Architecture / inputs | Amount | Cutoff | Mean fold recall | Mean fold precision | Caught cases | False alerts | Missed cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest / seven | SubTotal | 0.26 | 75.55% | 34.47% | 160 | 318 | 52 |
| Random Forest / seven | TotalDue | 0.29 | 71.78% | 36.31% | 152 | 277 | 60 |
| Histogram boosting / ten | SubTotal | 0.06 | 75.51% | 33.45% | 160 | 324 | 52 |
| Histogram boosting / ten | TotalDue | 0.08 | 73.17% | 34.51% | 155 | 300 | 57 |

Counts are summed account/snapshot cases across folds, not distinct identities.
Forest loses 3.77 percentage points of mean recall, misses eight previously
caught cases, and removes 41 false alerts. Boosting loses 2.35 points, newly
catches two cases but misses seven previously caught cases, and removes a net
24 false alerts. Forest recall falls in all five folds; boosting recall is
unchanged in two and falls in three. Mean fold average precision falls from
0.4722 to 0.4712 for forest and from 0.4827 to 0.4695 for boosting.

The shared-cutoff diagnostic shows why constrained selection matters. At 0.26,
TotalDue forest catches 159 cases with 317 false alerts, versus SubTotal's
160 / 318. At 0.06, TotalDue boosting catches 160 with 319 false alerts, versus
SubTotal's 160 / 324. However, TotalDue precision in fold 2 is 28.87% for forest
and 29.17% for boosting at these cutoffs, so neither shared-cutoff alternative
is eligible under the frozen rule. These diagnostics do not replace the
independently selected results.

At selected cutoffs, pooled recently active recall falls from 58.68% (71/121)
to 52.07% (63/121) for forest and 55.37% (67/121) for boosting. Active false
alerts fall from 256 to 216 for forest and 263 to 240 for boosting. Both retain
low active precision (21–23%). Individual-seed mean recall differences at each
variant's ensemble-selected cutoff are negative for all five seeds; these
diagnostics do not establish separate seed-specific precision feasibility.

Keep SubTotal under this experiment's recall-first objective: replacement with
TotalDue does not improve constrained recall for either architecture. This
does not test added tax/freight features, tuning, alternate floors, or new
independent cohorts. October outcomes were not accessed, no final model was
refitted, and no changes were promoted to `main`.

Commands (use a new comparison run ID):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/compare_subtotal_totaldue.py --preparation results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01 --run-id <new-comparison-run-id>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/verify_subtotal_totaldue_outputs.py --output results/debugging/subtotal_vs_totaldue/revenue-comparison-20261014-01
```

Saved reports include seed/ensemble probabilities, fold memberships/counts,
all threshold tradeoffs, selections, fold/pooled/subgroup metrics, seed metrics
at ensemble cutoffs, shared-cutoff diagnostics, matched decisions, and hashes.
The initial focused suite passed 37 tests (seven new runner tests plus existing
preparation, model smoke, selection, and metric checks). Two additional checks
for changed input hashes and modified saved metrics were then added; all nine
runner tests passed. No full development pipeline or full regression suite ran.
The saved-output verifier passed input/code/output hashes, fold separation and
coverage, seed averaging, recomputed cutoff grids/selections, all reported
fold/pooled/subgroup/seed metrics, shared-cutoff diagnostics, and paired decisions.

### Authorized retrospective October extension — 2026-10-15

The user requested evaluation on the final test set after the grouped revenue
comparison. Freeze all four configurations and their existing 30% development
rules: RF7 SubTotal 0.26, RF7 TotalDue 0.29, GB10 SubTotal 0.06, GB10 TotalDue
0.08. No new floors or cutoff search using October outcomes are included.
Refit each configuration on its matched 1,375 development rows with the same
parameters, balanced training weights, preprocessing, and five seeds (20 fits).

Use the saved October feature rows as the SubTotal baseline, and replace only
the two revenue columns for TotalDue using accepted orders strictly before
2013-10-01. Save protocol, models, scores, decisions, and a checkpoint before
loading the existing 492-account/46-churner labels. Reconcile historical source
amounts and reproduce the original GB10 and additional RF7 SubTotal predictions.
The entry point is `debug/evaluate_revenue_final_test.py`; it rejects existing
evaluation/model directories and never extracts outcome-window orders.

This authorization covers retrospective reuse of an already-inspected cohort,
not an independent final evaluation. Preserve all existing artifacts, disclose
prior inspection, report overall and recent-activity metrics, and do not select
new methodology or promote a replacement to `main` from these October results.

The completed extension is under
`results/final_evaluation/revenue-final-retrospective-20261015-02/`, with four
five-model bundles under `models/provisional/revenue-final-retrospective-20261015-02/`.
All 20 refits completed. The historical query returned 2,729 headers; 2,723
accepted orders were reconciled. SubTotal RF7 scores reproduce the previous
evaluation within 2.23e-16; SubTotal GB10 scores reproduce exactly.

| Model | Amount | Frozen cutoff | October recall | October precision | Caught | Missed | False alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF7 | SubTotal | 0.26 | 91.30% | 26.42% | 42 | 4 | 117 |
| RF7 | TotalDue | 0.29 | 89.13% | 28.67% | 41 | 5 | 102 |
| GB10 | SubTotal | 0.06 | 93.48% | 28.48% | 43 | 3 | 108 |
| GB10 | TotalDue | 0.08 | 93.48% | 30.07% | 43 | 3 | 100 |

TotalDue boosting catches the same churners and removes eight false alerts,
all from recently active accounts. Active recall remains 72.73% (8/11), and
active precision rises only from 7.55% to 8.16% (90 false alerts). TotalDue forest
newly catches one churner and newly misses two: net one fewer caught, with
15 fewer false alerts. Its active recall falls from 72.73% (8/11) to 63.64%
(7/11), with false alerts falling from 109 to 94; active precision remains
6.93%. Neither forest variant meets 30% overall test precision. TotalDue
boosting just exceeds 30% on this cohort; that is not a guarantee elsewhere.

These October findings differ from grouped development, where TotalDue reduced
recall for both models. Preserve both results and their populations. The
retrospective boosting improvement is descriptive evidence, not grounds for a
new independent-performance claim or automatic replacement on `main`.

The first attempt (`revenue-final-retrospective-20261015-01`) remains preserved.
It fitted 20 models but stopped before its label-loading stage because default
CSV parsing changed 15 GB10 baseline scores. A focused reproduction using the
same saved models showed round-trip feature parsing restored all original GB10
scores exactly. The corrected run uses round-trip parsing for scoring and saved
scores, with a regression test for serialized feature precision. No labels were
used to diagnose or repair this parsing issue and no settings/cutoffs changed.

The initial focused suite passed 24 tests. After the parsing correction, all
eight extension tests passed. The saved-output verifier passed input/code/output
and model hashes, checkpoint chronology, model-reload scores, seed means,
fixed-cutoff decisions, exact unchanged-feature parity, matched labels, and all
12 overall/subgroup metric rows. Syntax and diff checks passed; the full
regression suite and any independent final evaluation were not run.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/evaluate_revenue_final_test.py --preparation results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01 --comparison results/debugging/subtotal_vs_totaldue/revenue-comparison-20261014-01 --run-id <new-retrospective-run-id>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/verify_revenue_final_outputs.py --output results/final_evaluation/revenue-final-retrospective-20261015-02
```

## Nested grouped revenue study: protocol — 2026-10-15

Status: protocol recorded before implementation and execution; the study is now
complete. The user defines better
as higher recall without lower precision, or higher precision without lower
recall, at the 30% target floor. Compare SubTotal and TotalDue within RF7 and
GB10, preserving the prepared datasets, feature sets, model parameters, balanced
training weights, five model seeds, and native/training-only missing handling.
No October features, predictions, or outcomes enter this study.

Use three predefined split repeats (42, 78, 88), each with five outer and five
inner StratifiedGroupKFold splits by StoreID. Keep all reseller snapshots
together and use identical splits for both models and both revenue definitions.
Inner folds contain only outer-training identities. Select one cutoff per outer
fold/configuration using mean inner fold recall, precision >=0.30 in every inner
fold, and the existing tie rule/grid. Refit on all outer-training rows and score
the untouched outer fold with that cutoff. Do not retune using outer outcomes.
Each repeat requires 600 fits: four configurations × five outer folds ×
(five inner fits plus one outer refit) × five model seeds. Total: 1,800 fits.
Run one complete repeat first to check execution and measure runtime, then
continue the remaining predefined repeats without changing the protocol.

Report selected threshold and inner selection status for every outer fold,
alongside recall, precision, confusion counts, alert rate, ranking scores,
30% outer-floor feasibility, and recent-activity subgroup results. Infeasible
inner selections remain explicit with no fallback cutoff or alert decision.
Save their outer scores, but do not create a pooled operating metric from an
incomplete repeat. A pooled repeat report uses its five different frozen outer
cutoffs; record this as variable thresholds, not a single deployable threshold.
Report threshold ranges and medians only as descriptive stability diagnostics.

Summarize equal-weight means of repeat-level pooled recall/precision, retaining
per-repeat and per-fold metrics and floor failures. Estimate paired uncertainty
by resampling the 422 reseller identities with replacement, retaining all their
snapshots and repeat predictions together. Use 5,000 deterministic bootstrap
draws (seed 20261015), with no model refitting or threshold reselection in the
bootstrap. Report 95% descriptive intervals and conservative 98.75% marginal
intervals for the four metric differences (two models × recall/precision) as
an approximate multiplicity adjustment. These intervals condition on fitted
models/cutoffs; overlapping training sets and three split repeats do not supply
independent replications or measure full training-procedure uncertainty.

For a conservative development recommendation, require complete feasible inner
selections, pooled outer precision >=0.30 for both variants in every repeat,
nonnegative adjusted lower interval bounds for both metric differences, and
at least a one-percentage-point mean gain in recall or precision. The one-point
minimum is a practical default for this study, not an approved outreach-cost
target. Report any outer-fold floor failures even when pooled floors pass.
Apply the same symmetric criterion to SubTotal; otherwise classify the result
as tradeoff or inconclusive. Do not declare equivalence from a nonsignificant
result. Freeze a separate development cutoff only after this study; do not
average outer cutoffs into a deployment rule. Independent confirmation still
needs a new authorized holdout.

Use new directories under `results/debugging/subtotal_vs_totaldue/<run_id>/`,
with immutable protocol/input/code identities, separate per-repeat reports,
and a final analysis directory. Completed repeats are verified and preserved
when continuing; unfinished repeat directories are not overwritten. Retain
execution timing, individual/mean probabilities, memberships, inner cutoff
grids/selections, and all outer decisions.

### Completed nested grouped study — 2026-10-15

`debug/nested_revenue_comparison.py` completed all 1,800 fits under
`results/debugging/subtotal_vs_totaldue/nested-revenue-20261015-01/`.
The first repeat completed in 376 seconds and passed saved-output verification
before the other two frozen repeats ran (339 and 340 seconds). Completed repeat
files were preserved and checked on resume. All 60 inner selections were
feasible; no fallback cutoffs were used. October inputs/outcomes were excluded.

The following are equal-weight means of three repeat-level pooled operating
metrics. Each repeat pools decisions from its five independently selected outer
cutoffs; these are not metrics at one averaged threshold or on averaged-repeat
probabilities. Threshold ranges cover all 15 outer fits per configuration, and
medians are descriptive only.

| Model | Amount | Mean repeat pooled recall | Mean repeat pooled precision | Threshold range | Median threshold | Outer folds meeting 30% precision |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| RF7 | SubTotal | 73.11% | 34.53% | 0.18–0.35 | 0.28 | 14/15 |
| RF7 | TotalDue | 72.48% | 34.38% | 0.18–0.35 | 0.28 | 14/15 |
| GB10 | SubTotal | 75.63% | 33.47% | 0.04–0.23 | 0.06 | 13/15 |
| GB10 | TotalDue | 75.63% | 34.25% | 0.04–0.24 | 0.07 | 12/15 |

Every repeat's pooled precision exceeds 30%, but outer-fold failures remain.
Boosting's TotalDue recall difference is -3.30, +4.72, and -1.42 percentage
points in repeats 42, 78, and 88; the equal mean does not establish unchanged
recall in each split. Its precision difference is -0.36, +0.79, and +1.91 points.
Forest's recall difference is -0.94, 0.00, and -0.94 points, with precision
slightly lower for TotalDue in every repeat.

The paired intervals below concern TotalDue minus SubTotal, in percentage
points, with 5,000 reseller-cluster draws preserving snapshots and repeat
predictions together. They condition on the saved fitted models/cutoffs and
do not measure full retraining/reselection uncertainty. The 98.75% marginal
intervals are an approximate adjustment for four comparisons (two architectures
× two metrics); they are not independently calibrated joint confidence regions.

| Model | Metric difference | Mean difference | 95% interval | Adjusted 98.75% interval |
| --- | --- | ---: | --- | --- |
| RF7 | Recall | -0.63 | -1.41 to 0.00 | -1.64 to +0.29 |
| RF7 | Precision | -0.15 | -0.56 to +0.26 | -0.67 to +0.36 |
| GB10 | Recall | 0.00 | -1.80 to +1.76 | -2.30 to +2.23 |
| GB10 | Precision | +0.78 | +0.04 to +1.57 | -0.16 to +1.78 |

Neither architecture satisfies the frozen dominance criterion. Boosting has a
small observed precision benefit at equal mean recall, but it is below the
one-point practical minimum; uncertainty permits recall loss, and the adjusted
precision interval includes zero. Forest's point estimates favor SubTotal, but
the adjusted intervals do not establish strict superiority in both metrics.
Both conclusions are inconclusive under this criterion, not equivalence or
proof that any observed change is luck. There is no reliable development
evidence to replace SubTotal. No new final cutoff/model was selected or promoted.

For review, `report/outer_fold_results.csv` consolidates all 60 overall fold rows
with repeat, selected threshold, recall, precision, floor status, and confusion
counts; `report/outer_subgroup_results.csv` retains all 180 subgroup rows.
The report manifest hashes its source repeat files and exports. `analysis/`
contains all selections, threshold stability, repeat pooled metrics, paired
intervals, and comparison summary. Individual/mean inner/outer scores, every
inner threshold grid, memberships, and completion hashes remain per repeat.

All 16 focused tests passed, including nested identity isolation, precision
tie-breaking, infeasible selections, changed input/report safeguards, and paired
bootstrap checks. Duplicating identical repeats does not artificially narrow
the cluster intervals. `debug/verify_nested_revenue_outputs.py --require-complete`
passed all three repeats: input/code/output hashes, regenerated nested
memberships, complete seed coverage/means, independently reconstructed inner
confusion grids, selected cutoffs, held-out populations/decisions/metrics,
pooled results, threshold summaries, and reproduced bootstrap intervals.
Syntax and diff checks passed. The full regression suite and independent
holdout confirmation were not run.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/nested_revenue_comparison.py --preparation results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01 --run-id <new-nested-run-id> --first-repeat-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/nested_revenue_comparison.py --preparation results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01 --run-id <same-nested-run-id> --resume
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/verify_nested_revenue_outputs.py --output results/debugging/subtotal_vs_totaldue/nested-revenue-20261015-01 --require-complete
```

## Held-out SHAP explanations: protocol — 2026-10-16

Explain the ten-feature SubTotal boosting ensembles from nested repeat 42.
The nested study saved predictions rather than fitted outer models, so reproduce
its five outer ensembles (25 refits), check score parity, and preserve model
bundles separately. Explain only the 1,375 held-out development rows; no October
features or outcomes enter the analysis. Retain each fold's already-selected
cutoff and classify its examples without further selection or tuning.

SHAP 0.53.0, slicer 0.0.8, and tqdm 4.70.1 are optional research dependencies
installed under ignored `results/debugging/shap_analysis/runtime-shap-0.53.0/`.
The existing modeling environment is unchanged. A compatibility probe with
TreeExplainer probability output on eight native-missing GB10 rows failed
reconstruction (maximum error 0.01558); the precise library incompatibility was
not diagnosed. Those TreeExplainer attributions are excluded from reporting.

Use PermutationExplainer calling each seed model's `predict_proba` directly,
with 20 uniformly sampled rows from its outer-training fold as the shared
reference, sampling/permutation seed 42 + fold, and max_evals=210. Preserve
native missing values. This approximates probability-scale Shapley contributions;
verify every seed's baseline plus contributions reconstructs its prediction
within 1e-8, and that its baseline equals mean background predicted probability.
Average the five seed bases and contributions to explain the mean probability;
do not average raw log-odds explanations and transform them afterward.

The walkthrough is the highest-score recently active false alert (ties by
StoreID/snapshot). Also show the first StoreID/snapshot for caught churners,
missed churners, and false alerts. These purposeful examples are not a
representative performance sample. Make probability waterfall plots and held-out
summary/importance plots; report variation by fold and error category. Repeat the
walkthrough with max_evals=420 and permutation seed 78 to diagnose approximation
sensitivity. Small backgrounds, dependent features, and varying fold-specific
references limit interpretation; interventional masking can form implausible
feature combinations. SHAP is not evidence of causation, probability calibration,
or better model performance. Existing cutoffs and main remain unchanged.

### Completed held-out SHAP analysis — 2026-10-16

The run `results/debugging/shap_analysis/shap-gb10-heldout-20261016-01/`
reproduced and explained all 1,375 held-out account/snapshot cases from repeat
42. Its 25 refitted models are saved under
`models/provisional/shap-gb10-heldout-20261016-01/`. This is an explanation of
the existing validation models, not a new performance evaluation or a change
to the supported pipeline. The analysis completed in 225 seconds, with maximum
seed reconstruction error 5.56e-16. Reference backgrounds contain only training
identities, and every ensemble score matches the earlier nested-study score.

Start with reseller 752 at snapshot 2012-07-01, held out in outer fold 5. The
model score is 0.98616, above its frozen 0.06 alert cutoff, but actual churn is
zero: this is a false alert. The reference baseline is 0.25235, the mean model
prediction over the fold's 20 background cases, not the observed churn rate.
Approximate SHAP contributions explain the 73.38-percentage-point increase:

| Feature | Reseller value | Contribution to model score, percentage points |
| --- | ---: | ---: |
| Bike revenue share | 99.94% | +31.26 |
| Clothing revenue share | 0.064% | +8.51 |
| Days since last order | 62 | +7.22 |
| Recent weighted discount change | 0.34961 | +6.70 |
| Discounted line share, twelve months | 0.17391 | +5.21 |
| Twelve-month merchandise spending | 62,333.47 | +4.94 |
| Three-month merchandise spending | 3,304.44 | +4.73 |
| Recent weighted discount | 0.34961 | +3.84 |
| Accessory revenue share | 0% | +1.09 |
| Recent order count | 1 | -0.12 |

The largest term is the model's attribution to near-exclusive bike purchasing
relative to this reference. It does not establish that buying bikes causes
churn; category shares and spending are related, and feature masking mixes
their values. The high score is also not a validated 98.6% individual risk.
The waterfall explains a wrong decision, rather than making it correct.

Additional purposeful examples are saved with their original fold cutoffs:

| Example | StoreID | Snapshot | Score | Cutoff | Actual churn |
| --- | ---: | --- | ---: | ---: | ---: |
| Caught churner | 306 | 2012-07-01 | 0.84660 | 0.06 | 1 |
| Missed churner | 304 | 2013-04-01 | 0.03824 | 0.07 | 1 |
| False alert | 292 | 2012-07-01 | 0.71540 | 0.10 | 0 |

Across all held-out cases, mean absolute probability contributions rank recent
spending first (8.48 percentage points), bike share second (5.81), recency third
(5.04), annual spending fourth (4.82), and clothing share fifth (4.43). Recent
order count is small (0.08 points) in these fitted models. These quantities
measure attribution magnitude, not accuracy improvement or causal importance;
they do not justify dropping a feature without a separate controlled comparison.
Recent spending ranks first in four folds; bike share ranks first in fold 4.
Fold/error-category tables retain that variation. The pooled summary combines
different held-out models and fold-specific reference backgrounds, not one
deployment model. Cases can repeat reseller identities at different snapshots.

The walkthrough's largest attribution change with a different permutation seed
and doubled evaluation budget is 1.44 probability percentage points. This is
an approximation-sensitivity diagnostic, not a confidence interval, and does
not measure sensitivity to different background samples. Additive reconstruction
checks establish arithmetic consistency, not exact or causal attributions.

The run includes seed/ensemble contributions, feature values, example metadata,
background rows, reconstruction checks, importance by fold/error category,
sampling diagnostics, and plots. Start with
[the walkthrough waterfall](../results/debugging/shap_analysis/shap-gb10-heldout-20261016-01/plots/walkthrough_waterfall.png);
then inspect [global importance](../results/debugging/shap_analysis/shap-gb10-heldout-20261016-01/plots/global_importance.png)
and [the beeswarm](../results/debugging/shap_analysis/shap-gb10-heldout-20261016-01/plots/global_beeswarm.png).
The [manifest](../results/debugging/shap_analysis/shap-gb10-heldout-20261016-01/manifest.json)
records source/model/code/output hashes and runtime versions.

All four focused tests passed, including native-missing probability explanations,
incorrect reconstruction rejection, threshold equality, and deterministic case
selection. Saved-artifact verification passed score reproduction, seed means,
baseline/background agreement, per-seed and ensemble reconstruction, frozen
thresholds/error categories, and recomputed importance tables. The walkthrough,
global bar plot, and beeswarm were visually inspected. No October data or model
search was used. The full regression suite was not rerun.

Install the optional research packages without replacing existing model libraries:

```bash
bi_venv/bin/python -m pip install --target results/debugging/shap_analysis/runtime-shap-0.53.0 --no-deps shap==0.53.0 slicer==0.0.8 tqdm==4.70.1
PYTHONPATH=results/debugging/shap_analysis/runtime-shap-0.53.0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/explain_churn_shap.py --preparation results/debugging/subtotal_vs_totaldue/revenue-preparation-20261014-01 --nested results/debugging/subtotal_vs_totaldue/nested-revenue-20261015-01 --run-id <new-shap-run-id>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/verify_churn_shap_outputs.py --output results/debugging/shap_analysis/shap-gb10-heldout-20261016-01
```

Other dependencies use the existing project environment; the isolated install
report is retained alongside the research packages. New explanation/model
directories are required. Verification needs the saved models and base project
libraries, but does not import SHAP or refit. See the official
[PermutationExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html)
for the approximation and additivity properties.

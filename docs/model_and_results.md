# Frozen model, evaluations, and research findings

Supported development now uses nested grouped validation; see the
[pipeline guide](../pipeline/README.md). The fixed baseline and saved research
bundles below retain their recorded methodologies. Refactoring does not refit,
reselect or promote them. The period/class-weighted prediction adapter retains
the supplied development cutoff `0.27912545951329487` when explicitly requested;
actual May 2014 scoring awaits complete reconciled source inputs.


The final pipeline carries the methodology frozen before the original
2026-10-11 evaluation. Its source research commit is `f53202d` on
`pipeline_preparation`. Parameter/feature/seed changes require explicit new
methodology; probability-cutoff comparisons do not alter the trained model.

- Histogram gradient boosting: learning_rate=0.1, max_iter=100,
  max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0,
  early_stopping=False.
- Balanced training sample weights, native missing-value handling.
- Mean predicted probabilities across seeds 42, 78, 88, 1034, 2026.
- Ten features in the order recorded by the model manifest and input contract.
- Training snapshots July/October 2012 and January/April 2013: 1,375 rows,
  including 212 churn cases in the original dataset.
- Original primary cutoff: 0.06, selected from grouped development results.
  No model search or threshold selection is performed by the final pipeline.

The original test snapshot is 2013-10-01, with outcomes in
`[2013-10-01, 2014-04-01)`: 492 accounts, 46 churners. Predictions and fixed
alert decisions preceded outcome access. The original report remains under
`results/final_evaluation/gb10_grouped_20261011/` in the research checkout;
generated files and the original model remain outside Git.

| Cutoff | Original role | Recall | Precision | Caught | Missed | False alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | Secondary | 97.83% | 20.64% | 45 | 1 | 173 |
| 0.06 | Primary | 93.48% | 28.48% | 43 | 3 | 108 |
| 0.55 | Secondary | 71.74% | 56.90% | 33 | 13 | 25 |
| 0.88 | Secondary | 43.48% | 76.92% | 20 | 26 | 6 |

The primary rule fell below its intended 30% precision floor. Recently active
accounts had 8 caught churners, 3 misses, and 98 false alerts: 72.73% recall and
7.55% precision. Previously inactive accounts contained 35 of the 46 churners,
all caught by the primary rule. This subgroup mix limits interpretation of
overall recall as stable future-period performance.

The model remains provisional in terms of operational suitability. Merging a
tested implementation does not resolve alert-cost requirements, subgroup
weaknesses, probability calibration, or the source availability assumptions.
Changing cutoffs using these inspected outcomes requires a new independent
holdout to validate fresh performance. Preserve the original record and avoid
calling subsequent comparisons another untouched final test.


## Later research findings

Research completed on 2026-10-14 through 2026-10-16 is preserved separately on
`research/revenue-shap-results-20261016`, commit
`5cbd21f65912ef7d0f1b010eb585e18f950082a5`. Its
[protocols, implementation notes, and verification](https://github.com/Dzzz83/Reseller_Churn_Classification/blob/5cbd21f65912ef7d0f1b010eb585e18f950082a5/docs/debugging.md)
and [metric definitions and full results](https://github.com/Dzzz83/Reseller_Churn_Classification/blob/5cbd21f65912ef7d0f1b010eb585e18f950082a5/docs/evaluation_metrics.md)
are the detailed evidence record. The final pipeline continues to use SubTotal,
the same ten features, model settings, seeds, and original primary cutoff 0.06.
Experiment scripts are not imported into this implementation. Generated datasets,
models, predictions, and plots remain local ignored artifacts in the research
checkout; the published branch contains their protocols and verifier code.

### Retrospective October comparisons

The user authorized an additional seven-feature Random Forest evaluation and
then the SubTotal/TotalDue comparison. Both reuse the inspected October cohort
of 492 resellers and 46 churners. Each configuration applies its own cutoff
selected solely from grouped development under the 30% per-fold precision rule.
No October outcomes select settings or cutoffs. These are descriptive additional
comparisons, not new independent final tests.

| Model | Revenue amount | Development-selected cutoff | October recall | October precision | Caught | Missed | False alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest, seven features | SubTotal | 0.26 | 91.30% | 26.42% | 42 | 4 | 117 |
| Random Forest, seven features | TotalDue | 0.29 | 89.13% | 28.67% | 41 | 5 | 102 |
| Histogram boosting, ten features | SubTotal | 0.06 | 93.48% | 28.48% | 43 | 3 | 108 |
| Histogram boosting, ten features | TotalDue | 0.08 | 93.48% | 30.07% | 43 | 3 | 100 |

TotalDue boosting catches the same churners with eight fewer false alerts at its
higher cutoff. This does not isolate a benefit from TotalDue: retrospective
matched-cutoff checks catch 43 churners for both amounts, with SubTotal/TotalDue
false alerts of 108/109 at 0.06 and 99/100 at 0.08. These diagnostic cutoffs were
not used to replace the frozen original rule. TotalDue forest trades one fewer
caught churner for 15 fewer false alerts. Recently active precision remains low
for the TotalDue variants: 8.16% for boosting and 6.93% for forest.

See [the October extension record](https://github.com/Dzzz83/Reseller_Churn_Classification/blob/5cbd21f65912ef7d0f1b010eb585e18f950082a5/docs/debugging.md#authorized-retrospective-october-extension--2026-10-15)
for score reproduction, the CSV precision correction, and subgroup results.
The original SubTotal results above remain the original evaluation record.

### Repeated nested grouped validation

The study compares SubTotal with TotalDue for seven-feature forest and
ten-feature boosting, changing only three- and twelve-month revenue totals.
Category shares retain SubTotal denominators; all other features, populations,
labels, model settings, and model seeds match. October data is excluded.

Three predefined split seeds (42, 78, 88) each use five outer and five inner
StratifiedGroupKFold splits grouped by StoreID. All snapshots of a reseller stay
together. Inner folds select one cutoff per outer fold/configuration, maximizing
mean inner recall subject to precision >=30% in every inner fold. Outer resellers
are excluded from both training and cutoff selection. All 1,800 fits completed,
and all 60 inner selections were feasible.

| Model | Amount | Mean repeat pooled recall | Mean repeat pooled precision | Outer threshold range | Median threshold | Outer folds reaching 30% precision |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Random Forest, seven | SubTotal | 73.11% | 34.53% | 0.18–0.35 | 0.28 | 14/15 |
| Random Forest, seven | TotalDue | 72.48% | 34.38% | 0.18–0.35 | 0.28 | 14/15 |
| Histogram boosting, ten | SubTotal | 75.63% | 33.47% | 0.04–0.23 | 0.06 | 13/15 |
| Histogram boosting, ten | TotalDue | 75.63% | 34.25% | 0.04–0.24 | 0.07 | 12/15 |

Each metric averages three repeat-level pooled scores. Each repeat uses five
fold-specific frozen cutoffs; ranges and medians are descriptive, not deployment
cutoffs. Every repeat's pooled precision exceeds 30%, but individual outer folds
sometimes fail it. Grouped validation measures unseen reseller identities across
mixed periods; it is not an estimate of future-period performance.

Paired reseller-cluster bootstrapping uses 5,000 draws, keeping each reseller's
snapshots and repeat predictions together. Approximate multiplicity-adjusted
98.75% marginal intervals cover four metric comparisons. Boosting's TotalDue
recall difference is 0.00 percentage points, with interval -2.30 to +2.23;
precision improves by 0.78 points, with interval -0.16 to +1.78. Forest's recall
and precision differences are -0.63 and -0.15 points, with intervals -1.64 to
+0.29 and -0.67 to +0.36, respectively.

Neither variant establishes dominance under the frozen rule: no loss in either
metric within adjusted intervals, pooled precision >=30% in every repeat, and
at least a one-point mean gain in recall or precision. The practical gain minimum
is a research default, not an approved business target. Intervals condition on
fitted models/cutoffs; overlapping repeats are dependent and do not measure full
retraining/reselection uncertainty. Inconclusive evidence does not prove
equivalence or that observed differences are merely luck.

See [the full nested study](https://github.com/Dzzz83/Reseller_Churn_Classification/blob/5cbd21f65912ef7d0f1b010eb585e18f950082a5/docs/debugging.md#completed-nested-grouped-study--2026-10-15)
for exact threshold reports, per-repeat results, uncertainty, and verification.
No revenue-feature replacement or deployment cutoff was selected.

### Held-out SHAP explanations

The explanation study reproduced repeat 42's five outer SubTotal boosting
ensembles (25 refits) and explained all 1,375 held-out account/snapshot rows.
It uses approximate permutation SHAP directly on churn probability, with 20
reference rows sampled only from each outer-training fold. Five seed bases and
contributions are averaged on the same probability scale. A fast tree explainer
failed a reconstruction probe and was excluded. Optional SHAP packages stay in
the research environment; main has no SHAP dependency.

For reseller 752 on 2012-07-01, the model assigns 0.98616 against a cutoff of
0.06, but actual churn is zero: a false alert. The training-reference baseline
is 0.25235. Near-exclusive bike purchasing (99.94% bike revenue share) contributes
about +31.26 probability percentage points; the other features contribute a net
+42.12 points. SHAP explains the model's score, not whether that score is correct
or whether bike purchasing causes churn. The score is not a calibrated 98.6%
individual risk.

Across held-out cases, recent spending has the largest mean absolute attribution
(8.48 probability points), followed by bike share (5.81) and recency (5.04).
These are account/snapshot attribution magnitudes across fold-specific models
and references, not evidence of incremental feature usefulness. Correlated
features and interventional masking can create implausible feature combinations.
The largest walkthrough contribution change under a different permutation seed
and doubled evaluation budget is 1.44 probability points; this is a sensitivity
check, not a confidence interval or a background-sample robustness study.

Four focused tests and saved-artifact verification passed; maximum seed
reconstruction error is 5.56e-16. Reconstruction checks validate arithmetic,
not exact attribution, causation, calibration, or improved predictive accuracy.
See [the SHAP record and reproducibility instructions](https://github.com/Dzzz83/Reseller_Churn_Classification/blob/5cbd21f65912ef7d0f1b010eb585e18f950082a5/docs/debugging.md#completed-held-out-shap-analysis--2026-10-16).

### Current research continuation

The continuation, now on `research/development-recall-alerts-20261018`, uses
the stricter requirement of fewer false alerts while matching baseline recall on the same development
splits. The matched-history, breadth, cadence, learner/weight, product-mix and fixed
ensemble follow-ups do not produce an accepted replacement under their complete
frozen criteria. Descriptive ranking improvements do not establish operational
acceptance. See
[current metrics](evaluation_metrics.md#continued-matched-development-research--2026-10-16)
and the [continuation checkpoint](continuation_checkpoint.md). The original fixed
model and historical results above are preserved. Newly queried CompanyX source
availability limits are documented in [data analysis](data_analysis.md#live-source-availability-follow-up--2026-10-16).

The earlier 13-feature cadence/unit-revenue candidate completed the
explicitly authorized October test:45/46 caught with194 false alerts, versus
43/46 and108 for historical0.06. The matched-policy original10 catches46/46 with
183 false alerts. It fails the operational improvement objective; no replacement
or deployment follows. See [October test metrics](evaluation_metrics.md#authorized-latest-candidate-october-test--2026-10-17).

### Methods tried and current status — 2026-10-18

The baseline remains the operational reference; every candidate below remains
experimental and failed its complete frozen development acceptance requirements.
October results do not override those decisions.
The operational ten-feature model and original cutoff remain unchanged.
The full [20%–70% precision-target table](evaluation_metrics.md#authorized-all-candidate-precision-floor-evaluation--2026-10-18)
contains all 14 October-evaluated procedures, including failures and fallback.
These targets are development precision constraints, not numeric probability
cutoffs or guaranteed October precision.

| Procedure | What changed | Recorded lesson |
| --- | --- | --- |
| Original baseline | Ten-feature, balanced-weight HGB | Reference; keep historical 0.06 separate from new cutoff policies |
| Retained calibration | Logistic mapping on a disjoint training reference, retaining its fitted learner | High-recall mean FP fell 14.80%, but recall/subgroup gates failed |
| Smooth additive cadence | Training-fitted splines and logistic regression on ten features plus overdue ratio and gap variation | High-recall FP increased 10.81% |
| Reseller weighting | Inverse reseller-history influence followed by class balancing | High-recall FP increased 5.98%; equal account influence was not a solution |
| Activity experts | Separate HGB routes for recent-purchase and no-recent-purchase cases | Mean FP fell 4.05%, below the 5% practical requirement; other gates failed |
| Complementary blend | Equal mean of retained calibrated score and prior-corrected activity-expert score | High-recall mean FP fell 13.38%, but recall/subgroup gates failed |
| Period/class weighting | Fitting-only weight N/(8 × snapshot/class cell count), ten-feature HGB | High-recall FP increased 3.99%; later used as a controlled 40%-target comparator |
| RBF cadence | Training-only imputation/scaling, class-balanced RBF SVM, logistic transform of margin | High-recall FP increased 27.35%; margin transform is not fitted calibration |
| Early-quarter share | Add order share from [t−12m,t−9m) within annual history | High-recall FP increased 1.16% |
| Guarded component | Training-only component guard with baseline fallback | All 15 outer folds fell back; no improvement |
| Positive-floor veto | Remove baseline alerts below a retained-calibration floor learned from inner caught churners | Mean FP fell 8.04%, but baseline catches were lost |
| Earlier cadence/regularization | Original ten plus overdue ratio, annual count and gap CV; seven leaves, L2=1 | Favorable historical grouped means did not establish a replacement |
| Earlier cadence/unit revenue | Original ten plus overdue ratio, gap CV and latest-order subtotal/quantity | Passed later point/practical gates, failed FP uncertainty; October trade-off also failed |
| Period/class + personalized recency | Add only continuous overdue ratio to period/class HGB10 | Strongest recent ranking signal in latest cycle; full acceptance still failed |
| Period/class + annual product breadth | Add annual distinct ProductID count | Recent AP gain reversed in one repeat; not evaluated on October |
| Period/class + product density | Add annual distinct products / annual accepted orders | Ranking point gains, but more FP; not evaluated on October |
| Period/class + annual order exposure | Add only annual accepted-order count as a control | Weaker ranking gain and operating failure; not evaluated on October |

The high-recall percentages above refer to the continuation's matched policy:
maximize inner OOF precision subject to recall strictly above 90%. They must not
be compared as controlled effects against the later 40%-precision policy.
[Evaluation metrics](evaluation_metrics.md#development-only-autonomous-continuation--2026-10-18)
retains that table; the [journal](research_journal.md) preserves earlier feature,
learner, product-mix, bagging, projection and cutoff-transfer experiments as well.
The separate experimental-checkout feature-ablation study uses another operating
policy and is not a directly interchangeable headline comparison.

The latest personalized-recency ratio divides current recency by mean gaps
between distinct accepted purchase dates in [t−12m,t). Short undefined histories
remain NaN. The model may use its continuous values and interactions; it is not
an 'overdue > 1' rule. At the October 40% target it gives 38.74% precision and
93.48% recall, versus 45.16% and 91.30% for period/class alone. At the 60% target
it gives 77.14% precision and 58.70% recall, versus 77.78% and 45.65%.
There is no uniform dominance across these operating points.

The proposed business use is a low-cost support check-in, not an automatic
discount. Staff time, unwanted contact and follow-up capacity still matter.
No company-approved minimum precision or intervention-effect estimate is available;
40% is a research/presentation scenario, not a business requirement. The model
ranks purchase inactivity risk; whether outreach prevents churn needs a separate
intervention comparison. No ready-for-deployment claim follows from these scores.

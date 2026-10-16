# Why the grouped validation gain did not transfer to October

Completed on 2026-10-16 after explicit user authorization to execute the
diagnostic plan. This is exploratory explanation of already inspected outcomes,
not a new independent evaluation. The diagnostic calculations fit no models,
optimize no cutoffs, balance no datasets, and evaluate no additional candidate
or rule on October. A separate existing verifier performs five development-only
refits to audit saved-model reproducibility; these are not new candidate trials.
Research remains on `pipeline_preparation`; the operational pipeline and main
branch are unchanged.

## Findings

The candidate's broad operating behavior was similar across development and
October: it caught about 91% of active churners while incorrectly alerting on
about 38% of active non-churners. October had far fewer active churners, so
those false alarms dominated precision. Applying development subgroup recall
and false-positive rates to October subgroup counts predicts **20.17% overall
precision**, close to the actual **20.09%**.

This is an arithmetic explanation conditional on the saved fits and broad
subgroups, not causal proof that prevalence is the only change. The original
model performs better on October at its historical cutoff; the two models
face the same population. Development averages also conceal weaknesses across
snapshots, and the validation baseline cutoff policy differs from the
historical October baseline rule. There is no evidence here identifying one
new feature as the cause of the October failure.

## Audit of the experiment and comparison

- Saved prediction, evaluation-protocol, development and ablation inputs match
  their source manifest checksums. The model file matches the frozen SHA;
  its specification and cutoff match the saved evaluation protocol.
- Candidate and baseline predictions cover exactly the same 492 unique
  StoreID/snapshot pairs, with identical outcomes and activity values. All
  46 outcomes reconcile. Development predictions cover the same 1,375 rows
  separately within each repeat/configuration; repeats are not independent
  additional accounts. The ablation development feature table is identical.
- Independent evaluation verification reproduces candidate scores exactly,
  checks saved baseline scores, and reconciles all six subgroup metric rows.
  No score/label join or changed-cutoff explanation was found in these artifacts.
- Source review confirms historical feature windows, native missing values,
  training-only balanced class weights, five-seed averaging, and inner-only
  grouped selection. No imputation or resampling is introduced by this diagnosis.
  Saved latest-order dates precede October scoring; candidate feature columns
  have no infinite values. Four development snapshots mature their six-month
  labels by the October scoring boundary. These checks do not independently
  reconstruct every source feature or certify transaction-source completeness.
- The candidate adds three cadence features and changes maximum leaves from
  31 to seven and L2 from zero to one. Its cutoff is the frozen development
  value `0.21655335464238584`. The original October rule retains `0.06`.
- **The main development comparison uses inner-selected high-recall cutoffs
  for the original baseline.** That baseline has mean recall 91.51% and
  precision 27.16%. The same saved original scores at `0.06` instead have
  77.36% recall and 32.96% precision. Therefore the earlier development gain
  and the historical-rule October difference do not compare identical cutoff
  policies. Neither comparison alone isolates the feature/parameter effect.
- The adaptive selection procedure's 91.51% recall / 29.74% precision is not
  the performance of one fixed configuration. The fixed candidate configuration
  has grouped mean recall 94.65% and precision 29.31%, using separately fitted
  outer models and inner-selected cutoffs. October evaluates one full-data
  refit and its frozen cutoff, not those same outer models.

The 15 outer candidate cutoffs range from 0.0913 to 0.1700; the full-data frozen
cutoff is 0.2166. Different fitted models have different score scales, so this
does not prove a cutoff bug or justify selecting a new October cutoff.
Refit/cutoff-transfer stability remains a development-only follow-up hypothesis.

## Population and operating-point explanation

“Active” means at least one order in the preceding three months. Validation
values below are means across the three grouped repeats, except churn rates,
which describe their shared development population.

| Candidate, active resellers | Grouped validation | October |
| --- | ---: | ---: |
| Rows / churners | 1,221 / 121 | 447 / 11 |
| Churn rate | 9.91% | 2.46% |
| Recall | 91.18% | 90.91% |
| False-positive rate | 38.42% | 38.76% |
| Precision | 20.70% | 5.59% |

For each saved development repeat, expected active true alerts are October's
11 active churners times development active recall; expected false alerts are
October's 436 active non-churners times development active false-positive rate.
The same calculation uses 35 churners and ten non-churners for inactive accounts.
Precision is expected true alerts divided by all expected alerts. Calculate
within each repeat, then average; do not combine repeats as independent cases.

| Candidate, October subgroup counts with development operating rates | Expected | Actual October |
| --- | ---: | ---: |
| Active true alerts | 10.03 | 10 |
| Active false alerts | 167.53 | 169 |
| Inactive true alerts | 34.74 | 35 |
| Inactive false alerts | 9.74 | 10 |
| Overall precision | 20.17% | 20.09% |

This calculation keeps every actual dataset and every decision unchanged. It
is explanatory standardization using inspected outcomes, not model calibration,
an unbiased prospective forecast, or permission to tune to October's churn rate.

The inactive share of churners is 42.92% in development and 76.09% in October.
Both models catch all 35 inactive October churners with ten false alerts.
That easier-to-detect group makes October's overall recall look strong.

The original October rule's active false-positive rate is only 22.48%, compared
with the candidate's 38.76%. It catches eight active churners with 98 false
alerts; the candidate catches ten with 169 false alerts. Lower prevalence
therefore explains much of the candidate's validation-to-October precision
drop, but cannot by itself explain its disadvantage to the original rule.

## What development validation already reveals

The fixed candidate improves precision over the baseline with inner-selected
cutoffs in all 15 outer folds. Its pooled recall is above 90% in all three
repeats, but only 14/15 individual folds; the weakest fold is 78.57%. Fold and
repeat consistency at the aggregate level does not guarantee subgroup stability.

Snapshot breakdowns below use the same grouped out-of-fold predictions. They
are descriptive slices, not temporal validation or new splits. Differences
are candidate minus the high-recall baseline, averaged across repeats.

| Active development snapshot | Candidate recall | Candidate precision | Recall change | Precision change | False-positive-rate change |
| --- | ---: | ---: | ---: | ---: | ---: |
| July 2012 | 98.29% | 30.01% | +11.11 points | +3.93 points | -2.78 points |
| October 2012 | 84.44% | 11.82% | -13.33 points | -0.67 points | -2.87 points |
| January 2013 | 79.49% | 12.78% | +2.56 points | +4.42 points | -13.09 points |
| April 2013 | 90.74% | 22.58% | +8.64 points | -0.38 points | +7.44 points |

Active precision improves in eight of the 12 snapshot/repeat slices, rather
than all 12. October 2012 loses active recall; April 2013 trades additional
false alarms for higher active recall. January's active recall remains below
90%. These are development-observable limits of an overall recall constraint;
none establishes an additional subgroup requirement that the user had frozen.

## Existing development ablations

No new ablation was scored on October. Existing saved development predictions
were joined to the same feature table and summarized by behavior and snapshot.

| Development configuration | Active recall | Active precision | Active false-positive rate |
| --- | ---: | ---: | ---: |
| Original features/parameters, high-recall cutoff policy | 85.12% | 18.38% | 41.61% |
| Add cadence features, original parameters | 85.40% | 20.32% | 36.85% |
| Original features, seven leaves and L2=1 | 89.53% | 18.59% | 43.18% |
| Cadence plus seven leaves and L2=1 | 91.18% | 20.70% | 38.42% |

This supports the existing interpretation: cadence additions reduce false
alerts on average, while the regularization bundle primarily helps recall
and can increase active false alerts. It does not establish which change
caused October's failure. Removing seven leaves in the full context lowers
overall development recall by 5.19 points; it is not a proven free precision fix.

The prior family-adjusted precision contrasts still provide the clearest
evidence for gap variation and overdue ratio. Annual order count remains
uncertain, and L2's full-context adjusted interval nearly touches zero.
Snapshot/behavioral slices added here are post-hoc point estimates without
new significance claims, equivalence claims, or automatic feature removal.

## October error profiles: explanatory only

There are 81 added false alerts and ten removed false alerts, for a net 71.
All added false alerts concern active resellers with one recent-quarter order:

- 41 ordered one day before scoring, 15 ordered 32 days before, and 25 ordered
  62 days before.
- 67 had four annual orders and six had three: 73/81 had three or four annual
  orders. Median recent-quarter revenue is 4,321.22.
- Only eight have missing gap variation and two have missing overdue ratio.
  Missing cadence values therefore do not characterize most added false alerts.

Among active accounts with at least three annual orders, churn is 86/769
(11.18%) in development and 2/240 (0.83%) in October. This is a descriptive
change in the feature/outcome relationship; the October count is small and
does not prove a stable new relationship or a causal effect of annual count.

The two newly caught churners, StoreIDs 508 and 898, both ordered one day
before scoring, had two annual orders, and have missing gap variation.
Their recent revenues are 24.294 and 1,020.594. The remaining missed churner,
StoreID 1192, also ordered one day before scoring, had three annual orders,
and has recent revenue 713.796. Its candidate score is 0.1884, below the frozen
cutoff. These examples cannot establish a rule that reliably separates them.
No rule is proposed from these October patterns.

Overall AP improves from 0.7013 to 0.7892 on October, but active AP falls from
0.1553 to 0.0765. Aggregate ranking improvement therefore hides weaker active
ranking. Different subgroup prevalence also affects AP, so development-to-test
AP differences are not a pure measure of model degradation. The within-October
comparison uses matched populations but still contains only 11 active positives.

## Follow-up proposal, not executed

Freeze a development-only stability study before fitting further models:

1. Preserve reseller grouping and compare the same cutoff-selection policy
   for all methods. Keep the original `0.06` rule as a separately named reference.
2. Report overall and active/inactive precision, recall and false-positive rate,
   plus snapshot and behavior slices. Use all such slices consistently rather
   than selecting favorable ones. The primary target remains overall recall
   above 90%; a subgroup constraint would require an explicit revised objective.
3. Investigate cutoff/refit stability and the cadence/regularization tradeoff
   using development data only. Reuse the existing ablation evidence before
   proposing any additional search. Do not tune to October's observed prevalence
   or remove features on the strength of its error profiles.
4. Freeze any revised methodology before independent evaluation. A method
   informed by these October findings needs a fresh untouched evaluation cohort
   and explicit approval for an independent final performance claim.

## Reproduction, checks and limitations

Completed diagnostic artifacts are under
`results/frozen-candidate-failure-diagnosis-final/`: paired account cases,
cohort metrics, development stability and paired effects, ablation stability,
existing component effects, feature coverage, prevalence standardization, and
an identity manifest. Individual account outputs stay ignored.

```powershell
.venv/Scripts/python.exe debug/diagnose_frozen_failure.py --evaluation results/frozen-candidate-october-evaluation --development results/repeated-grouped-precision --features results/current-model-diagnosis/unlabeled_october_scores.csv --ablation results/grouped-component-ablation --output results/frozen-candidate-failure-diagnosis-NEW
.venv/Scripts/python.exe -m pytest -c comparisons/grouped_precision_pytest.ini debug/test_diagnose_frozen_failure.py -q
```

Use a new output directory. The diagnostic refuses mismatched populations,
outcomes, activity, artifact hashes, model identity and saved decisions.
Round-trip CSV float parsing preserves exact frozen-cutoff identity.

All **39 focused tests** pass, including 12 diagnostic tests for paired error
accounting, population/outcome/activity guards, prevalence arithmetic and exact
CSV cutoff parsing. Ruff, formatting and diff checks pass. Independent saved
study verification checks 540 development metric rows and inner/outer group
isolation; ablation verification checks 756 metric rows. Frozen October
verification reproduces scores exactly and checks six subgroup metric rows.
The grouped-study verifier additionally fits five deterministic copies using
development data only and compares their development predictions to the saved
bundle. It does not replace that bundle or score those copies on October.
These audit fits establish reproducibility, not additional model-quality evidence.

The normal `python -m pytest -q` suite remains blocked during collection by
the pre-existing `reseller_churn.config` module/package collision: the research
package lacks `TRAINING_SNAPSHOTS` imported by the final CLI. No orchestration
code changes, so no orchestration smoke run is needed for this diagnostic.

The baseline remains a documented reproduction; the exact original fitted
artifact is unavailable. Supplied outcomes reuse the earlier accepted-orders
reconstruction. No fresh transaction-database or outcome-coverage audit was
performed. Original source-data completeness and causal attribution remain
unresolved. October was already inspected; these findings cannot restore it
as an untouched test set.

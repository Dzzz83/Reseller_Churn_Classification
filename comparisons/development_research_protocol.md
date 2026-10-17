# Development research preparation — 2026-10-17

Status: evaluation/reporting preparation, NOT a frozen new candidate or permission
to fit one. No method is selected from October results. The operational model,
old research protocols and completed decisions remain unchanged.

## Objective and data boundary

Develop a model with fewer false alerts while preserving baseline recall. Use
only the approved 1,375 account/snapshot rows, 212 churn cases, 422 resellers at
2012-07-01, 2012-10-01, 2013-01-01, 2013-04-01. Keep every evaluation population natural.
StoreID/snapshot identify cases,not predictors. Histories [t-12months,t),eligibility
[t-6months,t),outcomes [t,t+6months); source recording-time/completeness assumptions
must be documented. Native missing features are allowed; future/outcome-derived
signals and unverified retrospective profile attributes are excluded.

October remains outside development: no test rows,labels,predictions,error profiles
or model-driven test diagnostics may choose a feature,parameter,cutoff or winner.
Development-only utilities do not import a final evaluator or construct test
labels. Any later test evaluation is a separate,explicitly authorized action
on a fully frozen method. Preparation includes no test evaluation.

## What has been decided

- Keep using development validation to diagnose mistakes and improve models.
  Repeated use is legitimate development. Record the searches/results so evidence
  strength is clear; do not turn a reporting label into the research objective.
- Compare original ten-feature baseline and candidates on identical rows,labels,group assignments
  and cutoff objectives. Original0.06 is a different historical operating policy.
- Keep all snapshots of a reseller together across inner/outer fitting boundaries.
  Fit weights,preprocessing,features,parameters and cutoff selection within training.
  Outer-error analysis may motivate later hypotheses,which remain development work.
- Report every repeat,fold,snapshot,activity group and snapshot/activity intersection.
  Show adverse changes and sparse/undefined rates prominently,alongside averages.
- Keep grouped and temporal evidence distinct. Temporal selection has already been
  tried and was weaker on points in the controlled April2013 comparison. It is not
  a proven fix; grouped time mixing is not an established cause of final failure.
- Measure cutoff transfer before choosing a remedy. No assumption that probability
  calibration or a higher numerical cutoff alone improves operational performance.

## Freeze before the NEXT candidate fit

Record in its immutable protocol:

1. Exact baseline/candidate feature formulas,source/code identities,HGB or other
   estimator parameters,training weights/missingness treatment,seeds and aggregation.
   No candidate has been supplied by this preparation document.
2. Exact evaluation case identities and reseller-disjoint outer/inner assignments.
   The existing grouped design is3 repeats (301, 919, 1201), 5 outer folds, 3 inner
   folds with seed 78. Reusing it is development reuse,not independent replication.
   If a temporal/grouped design is proposed,record its target population,training
   data and feasibility explicitly. All training labels must mature by the earliest
   validation prediction; grouping alone does not enforce this. Do not blindly
   exclude every future-cohort reseller and create survival-selected training data.
3. ONE common cutoff-selection objective. The existing research policy maximizes
   pooled inner-OOF precision subject to recall strictly >90%; ties recall then higher
   cutoff,scores>=cutoff alert,ties preserved. This does not guarantee baseline
   recall after refitting. Historical 30%-precision and fixed0.06 are separate policies.
4. Complete acceptance,including period safeguards,BEFORE results. Existing global
   requirements remain: recall>=baseline and FP<baseline EACH repeat; >=5% mean FP
   reduction; paired reseller-cluster recall lower>=0 and FP-reduction lower>0.
   Aggregate recall is pooled counts per repeat; reported means average repeats,
   not fold percentages or totals of repeated cases. Period limits apply separately
   to EACH repeat's pooled snapshot and activity comparisons,without averaging losses
   away. Snapshot/activity intersections and every fold remain reported diagnostics.
5. Explicit numerical period limits. Prospective RESEARCH DEFAULTS: at most 0.05
   absolute recall loss and ZERO additional false alerts in each snapshot/activity
   cell per repeat. These are conservative research choices,not business requirements
   or a retroactive test of old candidates. Freeze or justify alternatives before
   fitting. Undefined recall is not evaluable and cannot silently pass. Overall
   baseline recall preservation remains mandatory despite the subgroup tolerance.
6. Uncertainty plan: paired 5,000 reseller-cluster draws with seed 2026,carrying snapshots
   and repeats together; retain the exact interval tail and multiplicity/confirmation
   rule appropriate to the declared family BEFORE fitting. The prior last study used
   tail 0.05/56 and 50,000-draw confirmation only if all primary checks passed. Do not silently
   reuse that family denominator for an unrelated new family. Conditional intervals
   omit refitting/reselection and calendar shifts; seeds/repeats do not add resellers.
7. Success/failure/infeasibility reporting and the next action. Never tune an outer
   cutoff to enforce desired recall. Do not drop difficult cases or hide adverse
   periods. Code execution success is not model-quality acceptance.

## Prepared APIs and no-fit review

`comparisons/development_checks.py` validates approved cases,predictor identities,
reseller isolation,optional matured-label temporal/grouped splits and explicitly
supplied feature-event times. Matrix checks cannot verify historical source meaning.
The fixed-feature runner and shared partition/assessment paths call these guards.
Existing legacy temporal experiments retain their documented estimands; no automatic
switcher or new chronological cohort is created.

`comparisons/development_reporting.py` produces matched operating tables,worst-slice
flags and cutoff-transfer tables. It preserves undefined rates and score ties.
`assess_period_safeguards` takes EXPLICIT limits; it never picks limits from scores.
`evaluate_predictions(..., period_limits=...)` and
`run_fixed_hgb(..., period_limits=...)` apply/persist this optional prospective gate:

```python
period_limits = {"max_recall_loss": 0.05, "max_fp_increase": 0}
```

Omitting it retains legacy acceptance for historical callers; every new experiment
must explicitly freeze/pass its period limits. A period failure cannot be rescued
by favorable overall metrics or a different bootstrap. No old outputs are rewritten.

Read a saved fixed-numeric development study without fitting,model loading or
threshold selection:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.review_development_study \
  --source results/cadence-unit-revenue-20261017 \
  --output results/CHOOSE-A-NEW-development-review-directory
```

Only the declared `fixed_numeric_feature` development layout is supported; the
production CLI requires canonical snapshot counts and 422 resellers. Existing output
directories are refused. Output files:

- report.md: every repeat and every snapshot/repeat; no hidden period means.
- operating_metrics.csv: all folds/repeats,activity and snapshot intersections.
- paired_operating_metrics.csv,worst_slices.csv: matched changes,harm/undefined flags.
- cutoff_transfer.csv: saved cutoff on pooled inner OOF,each inner fold,and outer
  scores; training row/reseller ranges distinguish the different component fits.
- split_timing.csv: grouping and calendar-maturity description,not a temporal result.
- review.json,manifest.json: identities,limitations,and explicit no-fit/no-test status.

Cutoff transfer differences combine changing training size,membership,population
and inner selection. They do not isolate calibration or causal model instability.
A held-back cutoff-selection group or other remedy requires its own matched,
predeclared procedure and must account for reduced training data; none is adopted
here. Historical verification entrypoints that refit models are not no-fit reviews.

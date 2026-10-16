# Precision improvement under a grouped recall constraint

**Status after follow-up:** the development selection procedure improved
grouped precision, but the frozen candidate failed the October precision goal.
This document records that completed development experiment; it is not a claim
of improved independent final performance. Follow [the research rules](../docs/research_rules.md)
for subsequent experiments and [the failure diagnosis](frozen_failure_diagnosis.md)
for comparison-policy differences, population effects and subgroup instability.

The user-approved objective is to improve precision while maintaining recall
strictly above 90%, using grouped development validation only. Reseller IDs are
the grouping unit: every snapshot of one reseller stays in the same fold.
All work remains on `pipeline_preparation`. The operational frozen ten-feature
pipeline and main branch are unchanged.

## Data and validation

The input is the verified development-only feature CSV from the preceding
diagnosis: 1,375 rows, 422 resellers, and 212 churn cases across July/October
2012 and January/April 2013. The runner checks its preparation manifest and
checksum, rejects other snapshots and duplicate keys, preserves the original
training CSV parser, and preserves native missing discount/undefined ratio
values. It does not accept test inputs or construct final outcomes.

Three split seeds (42, 78, 88) each create five outer StratifiedGroupKFold folds.
Three inner grouped folds, seed 78, operate only on each outer-training set.
All candidates share folds and populations. The inner OOF predictions select
the highest-precision cutoff subject to pooled recall >90%. Ties favor recall,
then the higher cutoff. Configuration selection uses precision, recall, fewer
features, and stable configuration ID, in that order. Outer outcomes do not
select configurations or cutoffs.

## Frozen candidate grid

Four feature sets are crossed with two parameter settings:

| Feature set | Additions to the original ten |
| --- | --- |
| Original | None |
| Spending | Recent/previous-quarter revenue and order-count ratios |
| Cadence | Annual order count, overdue ratio, purchase-gap coefficient of variation |
| Full | Spending and cadence additions, plus last-order revenue / mean annual order value |

The baseline uses the original histogram boosting parameters. The regularized
setting changes only `max_leaf_nodes` from 31 to 7 and `l2_regularization` from
0 to 1. Learning rate 0.1, 100 iterations, minimum leaf size 20, disabled early
stopping, native missing handling, and balanced training-only weights remain
fixed. Both settings average the original five model seeds.

The complete grid and selection rules are saved to `protocol.json` before any
fit. The outer study performs 2,400 estimator fits. The synthetic smoke uses a
smaller grid and cannot establish model quality.

## Primary comparison and decision

The primary comparison is the adaptive inner configuration/cutoff selection
procedure versus the original ten-feature baseline with the same inner cutoff
rule. The earlier full-feature candidate is a secondary comparison. The
original 0.06 cutoff and individual grid configurations are diagnostic references.

Report each repetition's pooled recall and precision, every fold, active and
inactive subgroups, paired gained/lost churners, and removed/added false alerts.
Do not sum repeated predictions as independent accounts. A procedure is eligible
only if recall exceeds 90% in **every** repetition and mean precision exceeds
the baseline. Report failures rather than averaging them away.

Five thousand paired reseller-cluster bootstrap draws carry each reseller's
snapshots and repetition predictions together. The primary precision-difference
interval is marginal 95%, conditional on fitted models and selected cutoffs;
it excludes retraining/reselection uncertainty. Repetitions overlap. A positive
lower bound, in addition to eligibility, is the predeclared stronger evidence
criterion for freezing a research replacement. No practical improvement margin
has been supplied by the user, so numerical gains and alert counts are shown.

Only if that criterion passes does the same inner procedure select a final
configuration and cutoff on all development rows and save an ignored research
bundle. This does not deploy it or evaluate final testing. Otherwise no final
candidate is frozen. The outer scores of an individual grid configuration
cannot independently validate choosing that configuration after seeing them.

## Commands and artifacts

Use a new output directory for each run:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe -m pytest -c comparisons/grouped_precision_pytest.ini comparisons/test_grouped_precision_recall.py -q
.venv/Scripts/python.exe comparisons/grouped_precision_recall.py --smoke --workers 2 --output results/grouped-precision-smoke-NEW
.venv/Scripts/python.exe comparisons/grouped_precision_recall.py --development results/development-precision-diagnosis/development_features.csv --preparation-manifest results/development-precision-diagnosis/manifest.json --workers 4 --output results/repeated-grouped-precision-NEW
.venv/Scripts/python.exe comparisons/verify_grouped_precision_recall.py results/repeated-grouped-precision-NEW
```

The verifier independently recomputes inner cutoffs and configuration choices,
checks group separation and population joins, checks adaptive prediction
identity, reconciles confusion counts/metrics, verifies hashes, and writes the
local `report.md`. The run retains fold assignments, inner and outer predictions,
selection tables, estimator/training-content hashes, uncertainty, paired changes,
and the final decision under ignored `results/`.

At study launch, this checkout had duplicate TOML tables and committed conflict
markers in the package initializer. The user requested repair during execution;
both were fixed without changing the running standalone study. Full pytest now
gets past those errors but is blocked by a separate merged-code naming collision:
the research `reseller_churn.config` package shadows the final pipeline's
`config.py`, preventing import of `TRAINING_SNAPSHOTS`. Focused study tests use
the explicit comparison configuration and pass.

This remains exploratory development reuse. The earlier development partitions
and this feature family have been inspected. Grouped validation measures
held-out reseller generalization, without making a future-period performance
claim. Final testing stays outside selection. October is already inspected;
any new independent final claim requires a frozen procedure and an untouched
cohort with explicit approval.

## Completed results

The completed run is `results/repeated-grouped-precision/`. All 2,400 outer-study
estimator fits completed. The primary adaptive selection procedure passes the
predeclared repeat-level objective:

| Split seed | Baseline recall | Baseline precision | Adaptive recall | Adaptive precision | Net fewer false alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 91.98% | 26.90% | 91.51% | 29.48% | 66 |
| 78 | 90.57% | 27.87% | 92.92% | 29.62% | 29 |
| 88 | 91.98% | 26.71% | 90.09% | 30.13% | 92 |
| Mean | 91.51% | 27.16% | 91.51% | 29.74% | 62.33 |

Mean precision increases **2.59 percentage points**, with a paired cluster
95% interval of **+1.67 to +3.52 points**, conditional on the saved fits and
cutoffs. Mean recall is unchanged; individual caught cases differ. Each
repetition validates the same 1,375 rows once, so the means are not evidence
from 4,125 independent cases. No October testing metrics enter this comparison.

The result meets the pooled-per-repeat rule, not an every-fold guarantee:
11/15 adaptive outer folds exceed 90% recall, with a minimum fold recall of
71.43%. The weakest repetition is only 90.09%, so the recall margin is narrow.
The earlier full six-feature candidate reaches >90% in only two repetitions.
The adaptive procedure's configurations and cutoffs were selected inside
training; its reported result is not the performance of one fixed configuration.

The strong-support criterion passed. Full-development inner selection then
chose **cadence_regularized**: the original ten features plus overdue ratio,
annual order count, and purchase-gap variation, with seven leaves, L2=1, and
cutoff `0.21655335464238584`. The five-seed research ensemble is saved as
`research_candidate.joblib`, with specification in `frozen_candidate.json`.
That step adds 125 estimator fits (120 inner fits and five full-data refits).
It does not change the operational fixed model or access final testing.

The separate verifier checks all output hashes, population joins, group
separation, inner selection decisions, adaptive prediction identity, and 540
metric rows. It checks the frozen bundle against its file checksum, specification,
and deterministic refits using development features. In-memory joblib object
hashes are not assumed invariant across serialization; exact refit predictions
and the bundle file SHA establish the saved-bundle check. Refit predictions
are serialization/fidelity checks, not model-quality estimates.

All 14 focused study/diagnosis tests pass, as do Ruff and diff checks. The
synthetic smoke passes. The generated `report.md`, `verification.json`, and
`bundle_verification.json` provide the detailed local record. No final-test
evaluation, push, merge, or deployment was performed. This is encouraging
grouped development evidence, with the uncertainty and reuse limits above.

## Subsequent authorized October evaluation

After the grouped study and component ablations, the user requested evaluation
of the frozen candidate. It reaches 97.83% recall but only 20.09% precision on
the already inspected October cohort, with 179 false alerts versus 108 for the
original rule. Grouped development gains did not establish better October
precision at the saved cutoff. No model or cutoff is changed. See
[the separate evaluation record](frozen_october_evaluation.md); its descriptive
follow-up does not rewrite these development results or the original test record.

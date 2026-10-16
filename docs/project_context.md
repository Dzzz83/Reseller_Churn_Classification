# Project context

Updated against the grouped studies, ablations, October follow-up and failure
diagnosis on 2026-10-16. Current conclusions and future requirements are owned
by [the lessons and research rules](research_rules.md).
This document owns the project
overview and current research status. Database provenance belongs in
[data_analysis.md](data_analysis.md); feature definitions and measured evidence
belong in [feature_analysis.md](feature_analysis.md).
Metric explanations and completed-run results belong in
[evaluation_metrics.md](evaluation_metrics.md).

## Purpose and prediction contract

Predict whether an active reseller will place no orders during the next six
months. The current user-approved research objective is to **maximize precision
while maintaining pooled grouped recall strictly above 90% in every repetition**.
Feature, parameter, and cutoff choices use inner training folds; outer grouped
validation measures their performance. The older debug experiments used a 30%
minimum precision in every validation fold. A deployment alert budget has not
been established, and no research improvement has replaced the operational model.

For prediction snapshot `t`:

- Eligible reseller: at least one order in `[t - 6 months, t)`.
- Main feature history: orders in `[t - 12 months, t)`.
- Target `churn = 1`: no orders in `[t, t + 6 months)`; otherwise `0`.
- Both windows exclude their upper endpoint.
- Fit preprocessing, resampling, and models using training rows only within
  each fold. In the latest studies, inner OOF predictions select configurations
  and probability cutoffs; outer validation outcomes are excluded from selection.

The boundaries are implemented in
[prediction_window.py](../reseller_churn/data/prediction_window.py),
[eligibility.py](../reseller_churn/data/eligibility.py), and
[churn_labels.py](../reseller_churn/data/churn_labels.py).

## Repository structure

| Location | Responsibility |
| --- | --- |
| `reseller_churn/config/` | Paths, feature sets, seeds, model settings, and validation schedules |
| `reseller_churn/data/` | Loading, prediction windows, eligibility, labels, dataset assembly, temporal folds |
| `reseller_churn/features/` | Reusable main feature construction |
| `reseller_churn/modeling/` | Model factories, candidate definitions, training-only fitting, seed ensembles, and manifest-checked refitting |
| `reseller_churn/evaluation/` | Ranking and threshold metrics, precision-constrained selection, reports, and regression checks |
| `pipeline/` | Preparation and development entry points, plus historical numbered scripts |
| `comparisons/` | Separate validation and alternative-method experiments |
| `debug/` | Missed-churn diagnosis, CompanyX feature experiments, focused tests, and artifact verification |
| `datasets/` | Clean source CSVs and tracked processed regression references |
| `tests/` | Data-boundary, feature, model-baseline, and evaluation checks |
| `results/` | Ignored generated experiment reports |
| `models/provisional/` | Ignored provisional model objects |
| `docs/` | Project context, database analysis, feature analysis, evaluation metrics, and debugging findings |
| `.github/workflows/` | Regression tests, separate research audits, and bounded development smoke checks |
| `AGENTS.md` | Engineering and research rules for collaborators |

The working branch is `pipeline_preparation`. It must not be merged into `main`
automatically. The repository is a Python package defined by
[pyproject.toml](../pyproject.toml), requiring Python 3.11 or later.

The fixed scoring CLI coexists with research through explicit domain modules:
`config/frozen_model.py`, `data/preparation.py`, `model.py`, and
`evaluation/cutoff_evaluation.py`. There are no same-name module/package pairs.
Read [code architecture](code_architecture.md) for dependency ownership,
shared behavior, and artifact identity requirements.

## Development datasets and validation

The main labeled dataset has one row per eligible reseller and snapshot. A
reseller can occur at multiple snapshots, so rows are not independent identities.

| Snapshot | Eligible rows | Churners | Role |
| --- | ---: | ---: | --- |
| 2012-07-01 | 326 | 76 | Development |
| 2012-10-01 | 366 | 47 | Development |
| 2013-01-01 | 343 | 25 | Development |
| 2013-04-01 | 340 | 64 | Development |
| Total | 1,375 | 212 | Labeled development rows |

Counts are registered in
[validation_settings.py](../reseller_churn/config/validation_settings.py).

| Fold | Training snapshots | Validation snapshot |
| --- | --- | --- |
| 1 | 2012-07-01 | 2013-01-01 |
| 2 | 2012-07-01, 2012-10-01 | 2013-04-01 |

[TemporalDataset](../reseller_churn/data/temporal_dataset.py) checks that every
training label window ends by the validation snapshot. Chronological validation
asks about later periods. GroupKFold by `StoreID`, used in older comparisons,
and StratifiedGroupKFold ask about unseen reseller identities within mixed
historical periods. The earlier seven-versus-ten comparison used one grouped
split seed, 42. The latest precision study uses three repeat seeds (42, 78, 88),
five outer grouped folds, and three inner grouped folds on all 1,375 rows.
The user chose grouped validation as the criterion for this research; no temporal
validation or final testing is run by the latest studies. The earlier temporal
comparison scores 683 January/April rows and remains a separate historical
experiment. Grouped results do not establish future-period performance.

## Current model status

The latest repeated nested grouped procedure improves mean precision from
27.16% to 29.74% at unchanged mean recall 91.51%, with all three repetitions
above 90% recall. Its weakest repeat is 90.09%, and four individual outer folds
are below 90%; the requirement is pooled per repetition. Full-development inner
selection saves a 13-feature cadence/regularization research candidate with
seven leaves, L2=1, and cutoff `0.21655335464238584`.

The subsequent 14-configuration component ablation supports purchase-gap
variation and overdue ratio as precision contributors. Fewer leaves mainly
help observed recall; annual count and L2 precision contributions remain
uncertain after adjusting 17 comparisons. The ablation changes no saved model.
See [latest metrics](evaluation_metrics.md#grouped-precision-and-ablation-results--2026-10-16),
[study specification](../comparisons/grouped_precision_recall.md), and
[component evidence](../comparisons/grouped_ablation.md). All are exploratory
development reuse; final testing remains outside selection.

The user subsequently authorized scoring the unchanged frozen candidate on
the already inspected October cohort. It reaches 97.83% recall but only 20.09%
precision, catching 45 churners with 179 false alerts. The original rule catches
43 with 108 false alerts (93.48% / 28.48%). This fails the precision-improvement
objective on that cohort; the operational model is not replaced. See
[the descriptive evaluation](../comparisons/frozen_october_evaluation.md).

The development gains are not a proven replacement-model improvement. The
adaptive selection procedure (91.51% recall / 29.74% precision) and the fixed
configuration summary (94.65% / 29.31%) use the same development population
but evaluate different objects. Neither separately validates the single
full-data fitted bundle. The development baseline uses inner-selected high-recall
cutoffs, while the October historical baseline uses 0.06.

The [failure diagnosis](../comparisons/frozen_failure_diagnosis.md) finds similar
candidate active recall/FPR across development and October, with substantially
rarer active churn in October. Subgroup standardization closely reproduces its
precision drop; it does not explain away its disadvantage to the original model
on the same October accounts. Active precision already deteriorates in two
development snapshots. No individual new feature is established as the cause,
and source completeness remains unverified by a fresh audit.

Future research must follow the [research rules](research_rules.md): matched
cutoff policies, nested grouped selection, subgroup and false-positive-rate
reporting, development-only prevalence and refit-stability checks, and no
evaluation rebalancing or October-guided winner selection. The primary recall
criterion is unchanged; additional subgroup or practical-gain requirements must
be frozen before a new experiment. October is inspected, so an independent
claim for a revised method needs a fresh cohort and explicit evaluation approval.

The following describes the historical supported workflow, not the latest
standalone study's selection objective.

The supported development runner compares Logistic Regression, Random Forest,
and HistGradientBoosting across the registered feature sets and imbalance
strategies. It uses baseline parameters and five seeds by default; optional
bounded tuning is explicit. The per-fold recall-first selector requires each
fold's precision to meet the current sensitivity floor, then ranks eligible
thresholds by mean recall, worst-fold recall, mean precision, and stable ID.

The default 0.10–0.90 precision floors are sensitivity scenarios only. No
deployment minimum or operational winner has been approved. The separate debug
experiments fix the floor at 30%. Each runner floor writes a
separate selected or no-feasible manifest; selection scores are development
diagnostics, not independent performance estimates. See
[pipeline execution](../pipeline/README.md) for commands and output contracts.

Older numbered scripts and their 600-tree Random Forest / HistGradientBoosting
settings, single-seed provisional models, and thresholds 0.34/0.24 are retained
as historical diagnostics. They are not the configuration consumed by the new
selection and refit path. Model settings live in
[model_settings.py](../reseller_churn/config/model_settings.py).

### Earlier feature experiments and research direction

The [debugging record](debugging.md) documents the RFM missed-churn diagnosis,
the seven-feature improvement, CompanyX order-line extraction and 11 candidate
features, and the completed seven-versus-ten comparison. The latter tests three
architectures with balanced training weights under temporal and stratified
grouped validation: 12 primary configurations and 154 fits. Both methods are
retained; grouped scores do not replace temporal evidence.

Discounts help gradient boosting under both methods, while seven-feature Random
Forest retains the highest temporal mean recall in this comparison. Ten-feature
boosting is a candidate for further focused development, with the seven-feature
forest retained as a baseline. A model promotion or exclusive architecture
choice has not been implemented. See [earlier measured results](evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features).

The ten-feature set is defined locally in
[compare_seven_ten_features.py](../debug/compare_seven_ten_features.py). It is
not registered in `reseller_churn/config/feature_sets.py` or generated by the main
FeaturePipeline. The authorized final evaluation now saves a separate five-model
bundle under `models/provisional/gb10_grouped_20261011/ensemble.joblib`; this does
not register a new runner feature set or deploy the predictor.

## Separate experiments

- The [repeated grouped precision study](../comparisons/grouped_precision_recall.md)
  evaluates inner feature/parameter/cutoff selection under recall >90% and
  freezes a research candidate using development only.
- The [component ablation](../comparisons/grouped_ablation.md) isolates cadence
  additions and regularization, adjusts 17 precision contrasts, and leaves the
  candidate unchanged while recording simplification hypotheses.
- The [correlation-pruned audit](../comparisons/correlation_pruned_churn/README.md)
  reconstructs another method, with different feature windows, fold-local
  supervised pruning, and grouped/temporal validation. Its `full` window means
  all available past history, unlike the main pipeline's 12-month history.
- The [controlled strategy experiment](../comparisons/validation_strategy_experiment.md)
  gives both selectors the same April/October 2012 selection pool and evaluates
  on the same April 2013 development population. It retains a registered
  exclusion of `store_age`, uses partial April 2012 history, and discloses prior
  April 2013 outcome inspection. Its existing result is formally inconclusive
  under the predeclared winner rule. It does not replace the standard model
  settings automatically.
- Historical November 2013 results described in the alternative-method README
  are a previously inspected retrospective audit, not a fresh protected test.

## Protected evaluation

The final-test snapshot is 2013-10-01; its feature-only reference has 492 rows.
Its churn outcomes must remain outside development selection. The user
authorized the completed 2026-10-11 evaluation after freezing ten-feature
boosting, baseline parameters, balanced weights, five-seed averaging, and grouped
development cutoffs (30% primary; 20%, 40%, 50% secondary). The label window
ends exclusively at 2014-04-01 and contains 46 eligible churners.

The primary result is 93.48% recall but 28.48% precision, below its intended
30% level: 43 of 46 churners caught, three missed, and 108 false alarms.
Among recently active accounts, it catches eight of 11 churners but produces
98 false alarms. See [final metrics](evaluation_metrics.md#final-test-evaluation--2026-10-11)
for all four frozen rules and subgroup limitations.
The protocol and predictions were checkpointed before outcome
access; rules were not changed afterward. These October outcomes are now
inspected, so subsequent model changes need a new independent period for an
untouched final-performance claim. Existing code continues to exclude this
snapshot from development. The model remains provisional. Follow
[Stage 12 instructions](../pipeline/12_final_evaluation/README.md).

## Execution and maintenance

The supported runner prepares data and develops models. See
[pipeline/README.md](../pipeline/README.md) for commands, search options, and
report locations. Historical numbered scripts and separate research audits
run independently. Final evaluation remains outside the supported runner; its
authorized completed run is documented in [evaluation metrics](evaluation_metrics.md#final-test-evaluation--2026-10-11).

Development CI runs the regression suite before a bounded smoke check using
RFM features and one seed across all three architectures and imbalance
strategies. This checks execution and saved artifacts, rather than running the
full search on every change.

Tracked processed CSVs are fixed regression references. The supported
preparation runner checks generated engineered features, labeled development
data, and unlabeled final-test features against these references before writing
under `results/<run_id>/`. It records source row counts, schema, missing/duplicate
keys, and revenue reconciliation in `source_audit.json`. Legacy Stages 03 and 04
write to `results/legacy_preparation/`, not the tracked reference paths.

When changing research assumptions or feature inputs, retain the original
experiment definitions and results as evidence of their own configurations.
Report new evaluations separately. A successful smoke run establishes execution
correctness, not model quality.

# Project context

Reviewed against the working tree and final-evaluation artifacts on 2026-10-11. This document owns the project
overview and current research status. Database provenance belongs in
[data_analysis.md](data_analysis.md); feature definitions and measured evidence
belong in [feature_analysis.md](feature_analysis.md).
Metric explanations and completed-run results belong in
[evaluation_metrics.md](evaluation_metrics.md).

## Purpose and prediction contract

Predict whether an active reseller will place no orders during the next six
months. The business priority is churn recall, with precision and false alerts
reported alongside it. The current debugging experiments use an agreed **30%
minimum precision in every validation fold**. A deployment alert budget and
final operating requirement have not been established.

For prediction snapshot `t`:

- Eligible reseller: at least one order in `[t - 6 months, t)`.
- Main feature history: orders in `[t - 12 months, t)`.
- Target `churn = 1`: no orders in `[t, t + 6 months)`; otherwise `0`.
- Both windows exclude their upper endpoint.
- Fit preprocessing, resampling, and models using training rows only within
  each fold. Development validation rows score candidates and select their
  configurations and probability thresholds.

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

The working branch is `experimental_research`. It must not be merged into `main`
automatically. The repository is a Python package defined by
[pyproject.toml](../pyproject.toml), requiring Python 3.11 or later.

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
and StratifiedGroupKFold, used in the latest seven-versus-ten experiment, ask
about unseen reseller identities within mixed historical periods. The latest
grouped run uses five folds, shuffle=True, random_state=42, and all 1,375
development rows. Temporal validation scores 683 January/April rows and remains
the primary evidence for future predictions. Internal scores answer different
questions and cannot establish a universal winner between validation methods.

## Current model status

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

### Latest feature experiments and research direction

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
choice has not been implemented. See [latest measured results](evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features).

The ten-feature set is defined locally in
[compare_seven_ten_features.py](../debug/compare_seven_ten_features.py). It is
not registered in `reseller_churn/config/feature_sets.py` or generated by the main
FeaturePipeline. The authorized final evaluation now saves a separate five-model
bundle under `models/provisional/gb10_grouped_20261011/ensemble.joblib`; this does
not register a new runner feature set or deploy the predictor.

## Separate experiments

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

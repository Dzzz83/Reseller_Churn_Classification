# Controlled validation-strategy experiment

This registered retrospective protocol is separate from the latest seven-versus-ten
feature study in root `debug/`. It uses ordinary GroupKFold, a common April
evaluation population, and alert-budget selection. The latest study uses
StratifiedGroupKFold and a precision-floor threshold rule. See
[current results](../docs/evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features)
and [debug execution](../pipeline/README.md#debugging-and-feature-experiments).
The protocol below retains its original assumptions and settings.

On 2026-10-11, a separate user-authorized frozen-model evaluation accessed the
October 2013 test outcomes. That cohort is now inspected; this historical
comparison's own code, populations, and safeguards remain unchanged. See
[final-evaluation record](../docs/evaluation_metrics.md#final-test-evaluation--2026-10-11).

Run from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bi_venv/bin/python comparisons/validation_strategy_experiment.py
```

This is a retrospective comparison under an explicitly relaxed history-coverage
assumption. It leaves the standard pipeline and its feature formulas unchanged.
April 2012 has 306 observed days within the nominal 12-month window. April 2013
outcomes have been inspected in earlier development experiments, so this report
must not be called an untouched holdout or final-test estimate.

## Dates and information availability

| Snapshot | Purpose | Rows | Churners | Labels resolved |
| --- | --- | ---: | ---: | --- |
| 2012-04-01 | Selection pool / temporal training | 200 | 45 | 2012-10-01 |
| 2012-10-01 | Selection pool / temporal validation | 366 | 47 | 2013-04-01 |
| 2013-04-01 | Common development evaluation | 340 | 64 | 2013-10-01 |

Both selectors receive the same 566-row pool (411 unique resellers). GroupKFold
uses five StoreID-disjoint folds. Temporal validation trains on April 2012 and
validates on October 2012. GroupKFold selection is performed as of April 2013,
when both pool snapshots' labels are resolved; its mixed-period folds are not
presented as historical predictions at their individual snapshot dates.

Both chosen configurations refit on all 566 selection rows and score the same
April 2013 population. No April 2013 labels enter selection or fitting. There is
one chronological validation period; five random seeds do not add independent
time periods.

## Fixed candidate search and decision rule

- Architectures: baseline Logistic Regression, Random Forest (300 trees), and
  HistGradientBoosting, using existing factory defaults.
- Features: RFM, Pruned, Reduced, Full. The registered search excludes
  `store_age` from Reduced and Full, giving 3, 7, 13, and 16 predictors
  respectively. All inputs are sales-derived. This exclusion preserves the
  experiment's existing candidate definitions; it is not evidence of leakage.
  The project's [opening-year assumption](../docs/data_analysis.md#opening-year-assumption-and-provenance-limits)
  permits `store_age` as a historical feature. Adding it to this search would
  require a separately documented experiment and new evaluations.
- Imbalance: none, balanced training weights, training-only random oversampling.
- Seeds: 42, 78, 88, 1034, 2026. Deterministic logistic fits are reused across
  seeds; oversampled logistic fits use all five seeds.
- No exhaustive parameter search, supervised feature pruning, calibration, or
  inherited provisional thresholds.

Each candidate averages seed probabilities within a fold. At 10%, 20%, and 30%
budgets it alerts exactly `ceil(budget * rows)` resellers, sorting scores in
descending order and StoreID in ascending order to resolve ties. Its selection
score is the mean recall over these budgets and folds, with mean PR-AUC as the
first tie-breaker and simpler configurations as the second. These are diagnostic
budgets, not approved business capacity limits. There is no fitted probability
threshold. Internal validation scores from different strategies are not used to
declare the winner.

The common evaluation reports recall, precision, false positives, true positives,
F1, and average precision (the project's PR-AUC convention). At equal alert
counts, higher recall also means higher precision and fewer false positives.

A clear winner must outperform at least two common-evaluation budgets, lose at
none, have a 95% paired-bootstrap interval excluding zero for mean budget recall
difference, and show no reversed mean-recall result across the registered seeds.
PR-AUC and configuration-selection sensitivity are also reviewed before making
a research recommendation. Otherwise the result is inconclusive. The bootstrap
uses 5,000 paired reseller resamples with seed 2026. Its interval is conditional
on the selected models and this observed period; it does not quantify uncertainty
across future periods or eliminate prior development reuse of April 2013.

## Safeguards and outputs

Development orders are capped strictly before 2013-10-01. Feature construction
uses the existing half-open historical windows. Label construction is limited to
the three declared snapshots. All preprocessing and imbalance handling fit only
inside their training fold. Selection rejects any extra snapshot. Grouped folds
assert disjoint reseller identities; temporal training labels resolve by its
validation date.

The script writes checkpoints, a frozen selection manifest, and unlabeled common
predictions before revealing common evaluation labels. Complete reports go under
`results/validation_strategy_experiment/`, which is ignored by Git. Reference CSVs
and the guarded final-evaluation stage are not modified. Candidate metrics remain
selection diagnostics; per-seed and common metrics are saved separately.

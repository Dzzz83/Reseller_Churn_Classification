# Alternative Churn Method — Verification and Validation Comparison

Research branch: `research/churn-method-comparison`

Source: [CurrentTopicTest](https://github.com/pnn-re1506/CurrentTopicTest),
especially `6-6/reseller_churn_phase1_v2.ipynb` and
`6-6/reseller_churn_phase2.ipynb`.

This is an **independent audit and corrected reconstruction**, not a
modification of either original model pipeline. The active project models
remain unchanged.

## 1. Verified issues versus methodological choices

| Finding | Classification | Evidence / handling |
| --- | --- | --- |
| Raw CSV contains six invalid orders, but the original Phase 1 notebook removes them before modeling | Already correctly handled in original | Preserve the cleaned 3,800-order input and verify row counts |
| Source label tests `OrderDate <= snapshot + 6 months` | Confirmed inconsistent six-month endpoint | Use half-open `[t, t + 6 months)` |
| Five-fold GroupKFold groups by `StoreID` | Valid alternative, **not** leakage by itself | Retain as a separate unseen-reseller evaluation |
| Source feature pruning uses correlation with the labels of all development rows *before* GroupKFold | Confirmed supervised CV information leakage | Fit feature pruning inside each training fold |
| Source imputers, scalers and one-hot encoders are in an sklearn Pipeline | Correct train-only placement | Retain and test under both split strategies |
| `AnnualSales`, `BusinessType`, `Brands`, `Internet` and current store salesperson lack effective-from timestamps in the CSV | Historical validity **unverified** — not proven leakage | Keep optional diagnostic variant; do not present it as validated leakage-free |
| `GroupKFold` mixes snapshot periods and temporal validation does not | Different questions | Report separately, not as a universal winner |
| Source F1 threshold selected on OOF predictions, then evaluated on those same OOF labels | Potentially optimistic *CV threshold-dependent F1* | Development comparison uses fixed 0.5 diagnostic; threshold selection deferred |
| KMeans is fitted on the original November test features for segmentation | Descriptive post-hoc analysis, not classifier training | Excluded from predictive evaluation; do not use test-derived segments to tune model |

We have **not verified** whether the store attributes were recorded before
each historical snapshot. The static `stores.csv` does not provide
effective-from timestamps. Verifying this would require historical records
or original data documentation. Do not label these attributes as definitely
leaky or definitely safe without that evidence.

The original method's label endpoint changes their **2013-11-01** outcome from
26 churners to 35 under the half-open definition, with the same 475 eligible
resellers. That was confirmed from the original transaction CSV. Those
November labels have already been inspected and are **not** used in this
development runner.

## 2. Safeguarded development dates

The primary project's untouched test snapshot starts **2013-10-01**.
To avoid selecting models using labels past that date, this audit builds
only the following snapshots:

| Snapshot | Eligible | Churners | Last label boundary |
| --- | ---: | ---: | --- |
| 2012-05-01 | 200 | 70 | 2012-11-01 |
| 2012-08-01 | 393 | 74 | 2013-02-01 |
| 2012-11-01 | 347 | 28 | 2013-05-01 |
| 2013-02-01 | 343 | 29 | 2013-08-01 |

These counts were independently checked against the cleaned order CSV.
Both observation-window variants must have identical keys and labels.

### Strategy A: GroupKFold(StoreID)

Five folds. A StoreID is never present on both sides of the same fold.
All four development snapshots may occur on either side of the split.

**Question answered:** generalization to resellers not seen in training,
within the historical development population.

### Strategy B: chronological temporal validation

- Fold 1: Train 2012-05; validate 2012-11.
- Fold 2: Train 2012-05, 2012-08; validate 2013-02.

Every training label has fully resolved at its validation date.

**Question answered:** generalization to later historical periods.

Training population sizes and validation churn prevalence differ between
strategies. Score differences should not be attributed only to the
algorithm used to generate fold indices.

## 3. Feature scope

- `full`: all transactions **before** snapshot, for all-time feature
  quantities, plus explicitly windowed 12m/6m variables.
- `obs6`: only the preceding six months, with 6m/3m variables.
- `historical_only`: features constructed from prior orders plus
  `store_age` from YearOpened. Historical order territory and
  salesperson are derived from the last **past** order.
- `unverified_store_profile_diagnostic`: adds the original static store
  attributes (including AnnualSales and categorial store metadata).
  This variant is for isolating possible predictive value; it is
  **not** verified as safe for deployment.

The two models retain the original modeling architecture:

- RandomForest: 500 trees, min samples leaf 5, seed 42, no class weights.
- LogisticRegression: max_iter 5000.
- Numerical log1p/median impute (+ missing indicator)/scale.
- Non-log numerical median impute (+ missing indicator)/scale.
- Categorical most-frequent impute and one-hot encode.
- Numeric Spearman correlation filter |rho| > 0.8 fitted on fold
  training rows only. This is a reimplementation of the original
  rule, with validation labels excluded.

## 4. Run commands

From the repository root with the configured Python environment:

```bash
python -m pytest -q
python comparisons/correlation_pruned_churn/01_verify_data_and_labels.py
python comparisons/correlation_pruned_churn/02_verify_features.py
python comparisons/correlation_pruned_churn/03_compare_validation_strategies.py
python comparisons/correlation_pruned_churn/04_compare_with_baseline.py
```

Optional exploratory scope using unverified store attributes:

```bash
python comparisons/correlation_pruned_churn/03_compare_validation_strategies.py \
  --include-unverified-store-profile
```

This optional command **overwrites** the `03_` output files with
both historical-only and unverified-scope results. It does not alter
the core model or source datasets.

Outputs in `results/correlation_pruned_churn/`:

- `01_verified_snapshot_counts.csv`
- `03_fold_metrics.csv`
- `03_model_comparison.csv`
- `03_oof_predictions.csv`
- `04_matched_model_comparison.csv`

The baseline comparison runs the primary project's current seven-feature
Random Forest on the **same snapshot rows and labels** as the correlation-pruned method.
It retains the primary model's hyperparameters and uses seed 42; it is a
matched-dataset comparison, not an isolated feature-only comparison.

## 5. Original notebook reproduction (five training snapshots)

Run:

```bash
python comparisons/correlation_pruned_churn/05_reproduce_corrected_original.py
```

This is **not** the earlier four-snapshot development experiment. It keeps
the original notebook's five train snapshots (May 2012 through May 2013),
five GroupKFold(StoreID) folds, 500-tree RandomForest / LogisticRegression,
F1-selected OOF threshold, and November 2013 retrospective test.

Corrections are limited to the justified defects:
- Half-open six-month churn labels, including November's corrected 35 churners.
- Supervised correlation pruning inside each CV training fold.
- Cleaned order source, which **the original Phase 1 notebook also
  cleaned already**. The prior assertion that it trained with all 3,806 orders
  was wrong: the notebook removes the same six invalid rows.
- Historically unverified store-profile attributes are labeled **diagnostic
  only**, not silently certified as safe. The `historical_only` scope
  replaces original current store profile with prior-transaction-derived
  attributes. It is an adaptation, not an exact reproduction.

For the original-like store-profile diagnostic, log transforms include
`AnnualSales`, `AnnualRevenue`, and `NumberEmployees`, as in the notebook.

Reproduction files:
- `05_original_reproduction_metrics.csv`: pooled OOF and November results
- `05_original_reproduction_folds.csv`: per-fold metrics/selected features
- `05_original_reproduction_predictions.csv`: CV and November predictions
- `05_original_vs_corrected.csv`: original published versus corrected
  profile-diagnostic metrics; these remain **different experiments**

**Important comparability cautions:**
- OOF recall and F1 use the threshold that maximizes F1 on the **same**
  OOF outcomes, replicating the notebook, so CV threshold-dependent scores
  may be optimistic. No held-out label is used to choose thresholds.
- November has already been inspected and is therefore a retrospective
  audit, not an untouched holdout. Its label definition differs from the
  notebook's published test labels (35 rather than 26 churners).
- The protected October 2013 test snapshot and its outcomes remain untouched.
- Comparison with the previously published score is descriptive: the
  corrected run differs in label boundaries, fold-local pruning, and possibly
  feature availability. Do not attribute all differences to GroupKFold.

## 6. Interpretation safeguards

- PR-AUC is the **primary model-ranking metric**.
- Threshold 0.5 precision/recall/F1 is a **diagnostic**, not a
  tuned operating threshold.
- Report per-fold and aggregate results. Do not compare GroupKFold OOF
  results directly with temporal mean-fold PR-AUC as the same quantity.
- Feature selection and preprocessing are fit on each fold's training side.
- No calibration or threshold is selected using final-test labels.
- Repeated development comparisons can create selection optimism.
- The original method's November test has already been examined. It can later
  be used only as an explicitly labeled retrospective audit, not as a
  pristine unseen holdout.
- **No October 2013 final-test labels are evaluated by these scripts.**

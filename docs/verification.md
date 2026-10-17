# Final pipeline verification and review

Verified on 2026-10-11. Execution branch: `final-pipeline-preparation`.
Base: refreshed `origin/main`, commit
`b43e44d787236d77dc46fd9acc26b9ec831b3544`. Research source:
`pipeline_preparation`, commit `f53202dbe686a58b9f37b43d7da0b9f6667180c6`.
The research checkout, original scripts, and evaluation artifacts were preserved.

## Completed checks

| Check | Result |
| --- | --- |
| Final suite on Python 3.11.16, fresh dependencies | 44 passed |
| Final suite on original Python 3.14.7 environment | 44 passed |
| Synthetic standalone end-to-end smoke | Prepare, train, features, predict, and cutoff evaluation passed |
| Installed package outside the repository | All five CLI stages passed with synthetic CSVs and outcomes; imports resolved to site-packages |
| Development feature/label parity | All ten features and labels match 1,375 saved development rows, within 1e-10; parsed training feature values also match exactly |
| Original final feature parity | All ten features match 492 saved rows, within 1e-10 |
| Final prediction parity | All 492 serialized probabilities match exactly after round-trip parsing |
| Original cutoff decisions | All decisions match at 0.01, 0.06, 0.55, 0.88 |
| Existing final metric parity | All 12 overall/subgroup metric rows match within 1e-12 |
| Original evidence integrity | All 32 recorded original research/evaluation file hashes unchanged; original final verifier passes |
| Static checks | Ruff lint/format, Python 3.11 syntax, and workflow YAML passed |
| Repository checks | No research imports, generated outputs excluded, tracked datasets unchanged, diff and local Markdown links checked |

Parity training/scoring used the original Python 3.14 environment. Clean install
and smoke checks used Python 3.11.16 with NumPy 2.4.6, pandas 3.0.6,
scikit-learn 1.9.1, and joblib 1.6.0. Model loading requires matching saved
runtime versions and package hashes; cross-version model portability is not
assumed. Original artifacts are retained locally outside Git, with local parity
records under `results/parity/` in this checkout.

An initial parity check found differing scores for 15 accounts despite matching
aggregate metrics. Both ensembles gave identical predictions on the same
in-memory features. The difference was default CSV float parsing moving some
feature values across tree bin boundaries. Round-trip parsing restored the
original scores. Regression tests now cover scoring at a model boundary and
evaluation at a score exactly equal to a cutoff.

The saved October outcomes were reused solely to verify the extracted behavior
and reports. No protected outcome extraction, cutoff optimization, model
selection, or new held-out evaluation occurred. Synthetic smoke results are
execution evidence, not model-quality estimates. GitHub-hosted Actions have
not run during this local preparation.

## Main-branch scope

Included: final configuration, source reconciliation, historical features,
approved development preparation, fixed ensemble training/loading, ranked
prediction, explicit cutoff reports, focused tests, package installation, CI,
and final-purpose documentation.

Excluded: architecture/feature/imbalance searches, tuning grids, alternative
validation experiments, debugging investigations, dated experiment directory
dependencies, historical numbered runners, research-only tests and CI, raw
local exports, and generated model/report objects.

The order-header and order-line exports are explicit required inputs. Main's
tracked source CSVs do not contain all discount-feature facts. Source coverage
and independent evaluation chronology require user/source declarations; they
cannot be proven solely from the provided CSVs. Original order-time availability
and purchase/cleaning assumptions remain in effect.

## Suggested pull request

Title: Add final ten-feature churn pipeline with configurable cutoff evaluation

Body:

> Add a self-contained fixed-model pipeline that prepares historical features,
> trains the five-seed gradient-boosting ensemble, saves ranked probabilities,
> and reports user-supplied score cutoffs without retraining or threshold search.
> Research searches and diagnostics remain on pipeline_preparation.
>
> Verified 44 tests on Python 3.11 and 3.14, a fresh installed-package smoke,
> feature parity, exact parity for 492 original predictions, and all 12 original
> metric rows. Original research/evaluation artifacts remain unchanged.
>
> October outcomes are already inspected; further cutoff reports are
> descriptive. Original primary precision is 28.48%, below the intended 30%.
> Header/line CSV exports are required for discount features.

Review the diff against main and the documented CLI/input contracts. The user
will push the branch, create the PR, and decide whether to merge. No PR creation,
push, or merge is part of this preparation.


## Research-results documentation update — 2026-10-16

The main-branch update summarizes the inspected October comparisons, repeated
nested grouped revenue study, and held-out SHAP analysis in the existing
[model/results document](model_and_results.md#later-research-findings). The
separate research snapshot is commit `5cbd21f65912ef7d0f1b010eb585e18f950082a5`.
Its 41 focused research tests passed before publication, including the optional
SHAP smoke. These checks belong to the research implementation, not a new
main-pipeline model-quality claim. Recorded experiment verifiers and their
limitations remain in the linked research record.

The update is documentation-only: no changes to package code, model settings,
features, original cutoff, inputs, dependencies, tests, or workflows. Generated
models, raw extracts, account-level predictions, and plots are not added to Git.
The October comparisons remain retrospective; the nested study excludes October;
SHAP describes model behavior without claiming causation or prediction correctness.

Review checks for this documentation-only update: all 44 main-pipeline tests
passed on Python 3.14; reported results were checked against saved research
artifacts; all local documentation links/anchors and `git diff --check` passed.
Only README.md, docs/model_and_results.md, and this verification document change.
No training, evaluation, or independent holdout execution was performed for the
documentation update.

## Pipeline readability refactor — 2026-10-17

The refactor on `refactor/readable-pipeline` preserves the fixed methodology,
CLI options, CSV schemas, filenames, and manifest fields. Command handling now
uses explicit branches and source paths. Feature construction separates eligible
history, purchase summaries, and discount calculations. Training separates label
maturity checks from fitting, and evaluation uses descriptive decision/metric
names with explicit handling of undefined metrics. Modules retain their existing
responsibilities; no additional dependencies or application layers were added.

Python helper names changed: `date` became `parse_calendar_date`, `unique_ids`
became `require_unique_keys`, `finite_columns` became `require_finite_values`,
and CLI `parser` became `build_command_parser`. The source-loader endpoint is
now `end_date`. Python callers using those names or renamed
parameters must update their imports/keywords; terminal commands are unchanged.

Verification in the active Python 3.14 project environment:

- Baseline suite: 44 passed. Refactored suite: 49 passed, including new checks
  for no-alert/single-class metrics and rejection of incompatible package/runtime
  identities before joblib deserialization.
- Ruff lint and format checks passed. The standalone synthetic smoke passed
  across prepare, train, features, predict, and evaluate.
- A matched synthetic comparison against the original package produced
  byte-identical development data (900 rows), scoring features (210 rows), ranked
  predictions (210 rows, 206 distinct probabilities), cutoff metrics (18 rows),
  and account decisions (1,260 rows). Six cutoffs included 0, 1, and a cutoff
  derived from a saved score. Methodology and input hashes in manifests matched;
  code identities, output locations, and the resulting model-manifest hash were
  excluded from that metadata comparison.

Only synthetic inputs and outcomes were used for this refactor verification.
No protected outcomes were accessed, and no model-quality claim is made.
Package checksums intentionally change after refactoring, so existing model
bundles require their recorded code version or explicit retraining. Bundle
compatibility checks remain enforced.


## Feature-builder separation — 2026-10-17

Implemented on `refactor/feature-builders`, preserving the pre-existing working
files and unresolved `.gitignore`/`AGENTS.md` merge entries. Those conflicts were
not resolved or changed as part of this refactor.

Feature calculations and eligibility selection now live in `features.py`.
`build_purchase_features()` returns StoreID and the seven purchase features;
`build_discount_features()` independently returns StoreID and the three discount
features. `build_features()` prepares dated lines and combines the groups using a
validated one-to-one left join, adds the snapshot, and selects the fixed schema.
Temporary calculation columns do not escape the builders. Weighted discounts are
computed for the two required three-month periods, and annual discounted-line
share is calculated once; the unused combined-statistics container was removed.

Shared calendar-date parsing is in `dates.py`, preventing circular runtime
imports between source preparation and feature building. Shared account/snapshot
keys and the two feature groups are declared in `config.py`; the final ten-feature
order and model configuration are unchanged. Canonical Python imports are now
`reseller_churn.features.build_features`, the two builders in that module,
`reseller_churn.dates.parse_calendar_date`/`DateLike`, and
`reseller_churn.config.KEYS`. CLI commands, output files, feature formulas,
eligibility, and label rules are unchanged. Existing bundles still require their
recorded package version or explicit retraining because code hashes change.

Verification in the active Python 3.14 project environment:

- Baseline: 49 tests passed. Final suite: 54 passed.
- Five additional test cases cover independent builder schemas and unchanged
  inputs, gross-value-weighted discounts, date boundaries, zero versus missing
  discounts, retaining an eligible store when no discount row is returned, and
  empty eligible populations before/after synthetic activity.
- Exact comparison with pre-refactor synthetic outputs passed for 900 historical
  training samples and four snapshots (240 rows, 210 rows, and two empty outputs).
  Values, missing values, dtypes, row/column order, and CSV text all match.
- Ruff lint and format checks, CLI help, and standalone synthetic smoke passed.
  The smoke exercised preparation, training, feature building, prediction, and
  evaluation using synthetic inputs and outcomes only.

No real datasets were modified, no protected outcomes were read, and no new
model-quality evaluation was performed. Task-only baseline data and scripts were
removed after verification.


## Discount frequency naming — 2026-10-17

Implemented on `refactor/discount-frequency-name`, preserving the earlier
uncommitted readability and feature-builder refactors. Renamed
`discounted_line_share_12m` to `discount_frequency_12m` in the feature builder,
configuration, range validation, tests, and input contract. The builder includes
a short comment defining the annual fraction of discounted order lines. Formula,
feature position, missing-value handling, and model settings are unchanged.

Regenerated development data and four development snapshot feature tables under
`results/discount-frequency-rename-20261017/`. Preparation used
`datasets/orders_clean.csv`, `datasets/stores_clean.csv`, and the header/line
exports in `inputs/regeneration-20261016/`, with declared coverage
`[2011-05-31, 2013-10-01)`. Each output manifest records input hashes, package
hashes, runtime versions, and source reconciliation. Existing artifacts were
preserved; generated outputs remain outside Git.

Verification:

- `.venv/bin/python -m pytest -q`: 54 passed.
- Ruff lint and format checks passed for `reseller_churn`, `tests`, and `pipeline`.
- `.venv/bin/python pipeline/smoke_test.py`: passed using synthetic data only.
- Compared with preparation executed immediately before the rename: the
  development CSV is byte-identical after replacing only the column name. All
  1,375 samples, 422 stores, 212 churn labels, values, missing values, dtypes,
  keys, and row/column ordering match exactly.
- Separate feature outputs match the corresponding baseline development rows
  exactly: 326 rows for July 2012, 366 for October 2012, 343 for January 2013,
  and 340 for April 2013. All five regenerated output manifests verified.
- `comparison.json` and `baseline_manifest.json` in the output directory retain
  the comparison evidence and the pre-rename package/input identities. Temporary
  baseline files were removed.

No real-data model was retrained, no experiments were rerun, and no protected
test outcomes were read or evaluated. Old tables use the previous schema; old
model bundles require their recorded code or explicit retraining.

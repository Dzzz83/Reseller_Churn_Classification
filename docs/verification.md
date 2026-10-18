# Production pipeline verification and review

Implementation date: 2026-10-18. Branch: `feat/production-only-pipeline`, based on
inspected `main` at `7d896a8`. Production helpers were extracted selectively from
`refactor/grouped-pipeline-consolidation` at `3905f7d`; that branch was not merged.
The original research checkout and its uncommitted files remain in place. The
unrelated experimental worktree and tracked dataset references remain unchanged.

## Completed checks

Verification uses the active project environment: Python 3.14.8, NumPy 2.3.5,
pandas 3.0.1, scikit-learn 1.9.1 and joblib 1.6.0. Training and export checks
use synthetic data exclusively, with single-thread environment settings.

| Check | Result |
| --- | --- |
| Synthetic regression suite | 60 passed after restoring the public discount feature name |
| Source-checkout smoke | Grouped development, five-seed training, portable loading and Power BI export passed |
| Retained artifact loading | Matching SHA256, five fitted estimators, exact parameters/seeds/features/classes and matching runtime; no fitting or scoring |
| Wheel build and inspection | Built production-only wheel; packaged artifact manifest present; no research imports, binaries, CSVs, datasets or test packages |
| Isolated target installation | Installed wheel outside checkout; module CLI and installed entry-point help passed; synthetic smoke passed using installed package |
| Installed retained loader | Actual saved bundle validated outside checkout with packaged manifest and no adjacent research-file dependencies |
| Static/repository checks | Ruff and format passed; diff whitespace and operational Markdown links passed; tracked dataset references unchanged |

The suite covers source reconciliation, explicit exclusions, duplicate/invalid
keys and values, inclusive calendar windows, six-month eligibility,
label-availability protection, fitting-only period/class weights, reseller group
isolation, cutoff feasibility and equality, artifact incompatibility, exact CSV
schema, feature/score round trips, blank discounts, Yes/No flags, row uniqueness,
provenance, insufficient effective/loaded coverage and overwrite rejection.

A first synthetic fixture supplied an entirely missing discount-change column;
the pinned estimator raised a binning error during training. The execution
fixture now includes both present and missing values. No dependency or model
setting changed to bypass that error. Synthetic checks do not establish that
future fitting partitions with entirely missing columns will train successfully.

Implementation and conflict-resolution checks performed no real training,
scoring, prediction export, new performance calculation or access to protected
final-test labels, predictions, errors or performance artifacts. Loading the
retained model inspected only the bundle and production contract. The separate
user-authorized backup source check is recorded below. GitHub Actions passed on
the original production commit 8b8712b (run 39109765781); the resolved merge will
run the same synthetic, static and installed-package checks. A separate local
YAML parser check was not run because PyYAML is absent from the environment.

## Main-branch scope

Included: strict source validation and reconciliation, hash-bound source coverage
and approved exclusions, historical ten-feature preparation, development-only
labels, period/class-weighted five-seed fitting, nested grouped validation,
portable model loading and exact Power BI CSV export, operational documentation,
configuration, synthetic tests, packaging and CI.

Removed as active workflows: the old balanced-class model, historical primary
cutoffs, generic cutoff evaluation and research-results documentation. Research
code/results remain on their existing branches. There is one supported estimator
configuration; future grouped development is separate from saved-model export.

## Remaining export readiness

A separate user-authorized source check on 2026-10-18 inspected the supplied
CompanyX.bak and its matching CompanyX_Audit restoration. The backup checksum is
1a9f1457f96cf2dc00a4ea4bd8212b74f54337a727cd742722bfec4dd0591379;
its BackupSetGUID matches the restoration history. The original backup remained
mounted read-only and unchanged.

The full [2013-05-02, 2014-05-02) source window contains 1,812 orders and
30,790 line items. All order IDs, StoreIDs, dates, subtotals, quantities, line
counts and category revenues match the existing clean order file exactly.
No exclusions are required in this window. The backup independently reproduces
467 resellers with purchases in [2013-11-02, 2014-05-02); the pipeline continues
to calculate eligibility rather than hard-code that number.

The earlier detailed exports stopped at 2013-10-01. Replacement source files,
SQL evidence, backup/restore identity, a production source_coverage.json and
validation_summary.json are retained locally under the companion workspace's
inputs/snapshot-20140502-source-check-20261018 directory. They pass production
source and hash-bound coverage validation and remain outside Git. Completeness
is relative to the supplied backup; original recording-time availability cannot
be independently established from it. No churn labels or model scores were
constructed during the source check. The real export still awaits separate
execution authorization.

## Integration with current main

Conflict resolution integrates main at 241c6a0, including the purchase/discount
builder separation, shared calendar-date parser and adapted feature regression
tests from the intervening refactor. A follow-up restores the public feature
name discount_frequency_12m. The retained estimator's former name,
discounted_line_share_12m, is confined to its artifact contract and an explicit
inference mapping; no binary, feature value, score or cutoff changes. The
period/class-weighted production commands replace the old balanced-class
workflow. The suite now contains 60 passing synthetic tests, including retained-model
name adaptation and unchanged-score export coverage.

This follow-up prepares the user-authorized GitHub merge. No history rewrite,
binary upload, real model training or prediction export is included.

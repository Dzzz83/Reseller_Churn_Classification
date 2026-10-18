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
| Synthetic regression suite | 54 passed |
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

No real training, scoring, prediction export, new performance calculation,
database query or access to protected final-test labels, predictions, errors or
performance artifacts was performed. Loading the retained model inspected only
the bundle and production contract. GitHub-hosted Actions have not run locally;
CI mirrors the synthetic, static and package checks. A separate YAML parser
check was not run because PyYAML is absent from the active environment; the
workflow has not yet been executed by GitHub Actions.

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

For proposed snapshot 2014-05-02, independently establish complete reconciled
orders, headers and details for [2013-05-02, 2014-05-02), with valid reseller
references, source hashes, source-owner coverage attestations and independently
approved exclusions. The handoff records detail exports stopping at exclusive
2013-10-01. Orders through May 1 cannot establish detail completeness. Current
complete sources were not established or processed in this implementation.

The eligibility period is [2013-11-02, 2014-05-02). The proposed 467 eligible
resellers remains unverified and is not encoded as an expected population.
The real export awaits source reconciliation and separate execution authorization.
The model checksum/runtime checks pass; no model-side blocker or evidence
requiring retraining was found in the authorized artifact inspection.

Review the branch diff against main. No push, merge, history rewrite, binary
upload or real export is included in this preparation.

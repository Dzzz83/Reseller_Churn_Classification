# Pipeline refactoring plan and implementation record

The sections below record the 2026-10-10 refactor. The current grouped
consolidation is recorded at the end of this document. Preparation and model
development were originally refactored on 2026-10-10. Protected final
evaluation remains guarded. This document records the motivation, implemented
changes, and verification; current commands belong in the
[pipeline guide](../pipeline/README.md).

## Why the workflow changed

The previous numbered scripts made separate choices: architecture and feature
comparisons favored average precision, while threshold selection favored F1.
Later training consumed preset settings rather than automatically using those
comparison results. This did not consistently apply the project's recall
priority.

The scripts also repeated fitting and reporting, wrote generated data to
tracked reference paths, and evaluated five-seed ensembles while saving
single-seed provisional models.

## Implemented changes

| Area | Change |
| --- | --- |
| Orchestration | The supported runner calls preparation and model development. Historical numbered scripts remain available separately. |
| Preparation | Source audit, unchanged feature/label definitions, and reference parity checks precede saving generated data under a unique run ID. |
| Comparison | Complete configurations combine architecture, feature set, imbalance strategy, parameters, and seeds. The search is bounded, with optional explicit tuning. |
| Selection | Recall determines the winner for each requested minimum precision. Each validation fold must meet the requirement. Infeasible settings stay explicit. |
| Prediction reuse | Candidate probabilities are reused for threshold comparisons, precision experiments, plots, and error reports. |
| Refit and persistence | Each distinct selected configuration is refit once. Its saved ensemble uses the same seed-averaged prediction policy as validation. Multiple precision settings can share one bundle. |
| Traceability | Selection manifests record configuration, threshold, data identity, folds, seeds, and repository status. Loading checks model metadata and checksums; refitting verifies actual data identity. |
| CI | Regression tests run before a bounded development smoke check. Generated reports and model bundles are retained as CI artifacts. |

The selected probability threshold is stored in the manifest. A model bundle
holds the fitted ensemble; callers apply the threshold from the corresponding
manifest. Final evaluation was outside this refactor. A separate guarded
entry point in `debug/` subsequently completed the authorized October test on
2026-10-11; it remains outside the supported runner. See
[Stage 12](../pipeline/12_final_evaluation/README.md) for its frozen methodology,
artifacts, and restrictions on reusing the inspected test.

## Where to read the current behavior

- [Project context](project_context.md): prediction contract, data roles,
  temporal folds, research status, and repository responsibilities.
- [Pipeline guide](../pipeline/README.md): entry points, search options,
  generated files, and historical commands.
- [Feature analysis](feature_analysis.md): candidate sets, feature formulas,
  and historical feature evidence.
- [Evaluation metrics](evaluation_metrics.md): recall selection, precision
  requirements, metric interpretation, and historical measured results.

The controlled validation-strategy experiment retained its registered protocol.
Tracked processed datasets remain regression references. Historical results
retain their original settings and must be distinguished from new selections.

## Verification

The checks for the original refactor completed on 2026-10-10; these are recorded
historical checks, not a newly executed suite on every subsequent experiment:

- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python -m pytest -q`:
  **67 tests passed**, including the final independent reviewer run.
- `bi_venv/bin/python -m compileall -q pipeline reseller_churn tests`: passed.
- `git diff --check`: passed.
- A bounded real-data smoke used RFM features, three architectures, three
  imbalance strategies, seed 42, and two temporal folds: **18 validation fits**
  across nine configurations. It exercised all nine default precision minimums.
- Preparation matched the tracked references and passed all ten source-audit
  checks. All nine manifests and five distinct refitted bundles reloaded.
  The three tracked processed-reference SHA-256 values were unchanged.

Tests cover historical prediction compatibility, training-only weighting and
resampling, recall selection, infeasible precision settings, stable ties,
changed-data rejection, JSON round trips, and saved predictor consistency.
Final review also fixed threshold-ID collisions for cutoffs such as 0.401 and
0.404 and made incomplete model-bundle references fail with a clear error.

The smoke artifacts are in
[`results/codex-luna-verification-20261010-1/`](../results/codex-luna-verification-20261010-1/).
They establish execution and artifact consistency. The full five-seed search,
optional tuning search, and protected final evaluation were not run, so the
refactor does not establish a better model or higher recall.

## Subsequent debug experiments

The later feature-discovery and seven-versus-ten studies run independently in
root `debug/` and write under `results/debugging/`. They use the existing model
factory and recall selection helpers, but do not register a ten-feature set
in the supported runner or promote a new model. Their completed-run evidence
and 39 focused checks are in [debugging.md](debugging.md), with current results
in [evaluation_metrics.md](evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features)
and execution prerequisites in [pipeline/README.md](../pipeline/README.md#debugging-and-feature-experiments).
The 39 focused checks and the earlier 67-test refactor suite have different
scopes; neither count describes a new full-suite run for this documentation
update.


## Grouped pipeline consolidation — 2026-10-18

The implementation branch preserves the current development research in a Git
checkpoint and consolidates it with `pipeline_preparation`. The original dirty
checkout is unchanged. No branch is merged or pushed automatically.

Supported development now uses repeated nested reseller groups. The package
owns reusable split validation, explicit cutoff policies, period/class weights,
ensemble aggregation and bundle adapters. Thin pipeline scripts share command
arguments and preparation code with the installed CLI. Temporal/reference-parity
orchestration remains in named legacy modules for historical reproduction.

Source fixes enforce effective loaded coverage, independently approved rejected
order IDs, nonnegative category revenue and midnight dates. Imported ten-feature
matrices receive eligibility/revenue/share consistency checks. Saved-model export
validates model and feature provenance and applies a supplied full-precision
threshold without fitting or reading outcomes. Old pickle class locations and
original manifests remain intact; compatibility is explicit per bundle format.

Synthetic tests protect grouped separation at both levels, inner-only choices,
infeasible partitions, weighting, serialization, float/cutoff boundaries and the
source failure cases. One CI workflow replaces the overlapping real-data and
temporal development jobs. Current commands belong in `pipeline/README.md`;
verification results belong in `docs/verification.md`.

The planned May 2014 real prediction export remains separate. Neither recorded
last-purchase dates nor pre-October detailed exports establish all required
source coverage or historical recording-time availability.

# Final pipeline verification and review

For the subsequent repair that lets the research and fixed workflows coexist
in this checkout, see [code architecture and completed local checks](code_architecture.md#completed-local-checks--2026-10-16).
That repair passed 191 combined tests; the narrower historical checks below
describe the original final-pipeline preparation and its separate code state.

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


## Authorized veto October evaluation — 2026-10-18

Eight focused tests passed in0.77s, including synthetic retained classifier/map
reconstruction, floor/tie/subset guards, required prediction checkpoint,
datetime/string outcome join and complete no-fit outcome reporting smoke.
Ruff check/format checks passed for the four evaluation files;git diff --check
passed. Independent audit verified40training identities,1476policy decisions and
all9overall/activity metric rows. Preparation and completed-outcome manifests
verified. Reporting repairs preserved frozen predictions and partial artifacts;
no refitting or cutoff revision. The checksum-bound completion record is
results/positive-floor-october-evaluation-20261018/completion_manifest.json.
Full results/limitations:the matching summary.md. No operational change/deployment.


## All-candidate precision-target evaluation — 2026-10-18

21 focused tests passed in7.51s; two existing SVC deprecation warnings. Coverage
includes all six learner-retention/serialization paths, historical window/quarter
boundaries, missing outcomes, exact precision floors, intact ties, infeasibility
and the complete13candidate inventory. Ruff check/format checks pass for seven
task files;git diff --check passes. Independently reproduced117developmentcutoff
choices,128logical fitting identities,7full modelbundles and276test confusion rows.
92targets feasible,25infeasible;all retained. Root and evaluation manifests verified;
late report/checkpoint/audits bound in completion_manifest.json. Legacy runtime and
Windowsmanifest adapters and interrupted-fit recovery are documented without model
or target changes. No test-driven selection or deployment. Authoritative output:
results/all-candidate-precision-october-20261018-04/report.md.


## Development ranking-information experiments — 2026-10-18

Eight focused tests pass:featureprovenance/invalidhistory,nested synthetic model
reload/cutoffisolation,weighted tied AP/AUC vs sklearn,existing precisionfloor/
periodweights. Ruff check/format andgit diff --check pass. Fourfixedarms each300
HGBfits;independentno-fit verifier reloadsall1200models,reproduces120innercutoffs,
2160operatingrows andfull5000operatingbootstrapdrawsperarm. Allrecentpoint ranking
gates recomputed;17rankingbootstrapdraws independentlyspotcheckedperarm. Allfour
fullacceptance decisionsFAIL,no50k/exactrescue;recencyhasprovisionalrankinggain.
NoOctoberinputs,SQL,newlabels,operationalmodificationorpush. Consolidatedmanifest
verified:results/development-ranking-information-20261018/manifest.json.

## Personalized-recency authorized October table — 2026-10-18

8focused tests pass in4.59s (nested synthetic persistence, feature provenance,
weighted ranks, precision cutoffs, period weights). Ruff check/format and
 git diff --check pass. Independent saved-artifact audit reloads20models,
checks20fitting weight vectors and fixed model configurations, reproduces
6cutoff choices and18overall/activity confusion rows with sklearn. Both
prediction-stage and evaluation manifests verified. Updated comparison table
preserves all13previous rows and adds one. No operational code change or push.
Audit/results:results/personalized-recency-october-20261018/.

## Research documentation reconciliation — 2026-10-18

Documentation-only update from completed saved reports. The full 14-procedure
20%–70% October precision/recall table is embedded in evaluation_metrics.md;
new target TP/FP/FN/TN and all three personalized-recency validation repeats are
included. README/model summary/checkpoint now identify the latest method and
preserve all failed acceptance decisions. Feature analysis and research rules
record warning overlap, provisional ranking evidence and unresolved attribution.

Verified all 84 precision/recall table cells against saved combined metrics,
six new target confusion rows, three validation-repeat rows and four development
mean/interval rows against saved artifacts. Local Markdown files and heading
links resolve after repairing two historical seven-versus-ten anchors.
`git diff --check` passes. No new model fitting, final evaluation or behavior
tests ran for these static edits. Prior execution checks above remain historical
checks, not newly rerun claims.


## Grouped consolidation and pipeline fixes — 2026-10-18

Implementation took place in an isolated worktree on
`refactor/grouped-pipeline-consolidation`, based on research checkpoint
`4bb611bd6cd27d1803434ec112d85cafa37771a2`. The original checkout, index and
uncommitted research files were preserved. Changes are separate from historical
research results; no real-data model was trained or scored and no protected
outcomes, saved predictions, errors or performance reports were used.

With `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and the existing project `.venv`:

- `python -B -m pytest -q`: **99 passed**. The default suite now uses synthetic
  grouped/core cases; historical and reference-data tests remain explicit-path
  checks. It includes source validation failures, loaded coverage, exclusions,
  grouped isolation, inner-only selection, infeasible policies, model reloads,
  architecture compatibility, exact cutoff equality and float round trips. Final
  review also verifies every retained estimator setting, including class_weight,
  against frozen constructor defaults.
- An explicit compatibility run covering grouped research, fixed-feature
  comparisons, period/reseller weights, precision-floor selection, retained
  synthetic candidates, candidate development, legacy preparation/workflow and
  synthetic boundary safeguards: **46 passed**. One existing scikit-learn SVC
  `probability` deprecation warning remains in the historical RBF candidate.
- `python -B pipeline/grouped_smoke_test.py`: passed preparation, nested grouped
  fitting, serialization, feature construction and outcome-free score/flag
  export, using disposable synthetic files only.
- Ruff import/lint and formatting checks passed for `reseller_churn/`, supported
  pipeline entry points, grouped/source-contract tests and the three modified
  grouped comparison modules. `git diff --check` passed.
- The actual saved `period_class_weighted/frozen_models.joblib` loaded through
  the new adapter with five estimators and matching checksum
  `6ee382620eeae80bb2fce68734813e0323b7620a56c4b73f867ecfdad66894ba`.
  No feature table was loaded and no prediction method was called for it.

After installing `setuptools==84.0.0` and `wheel==0.48.0` in the existing
project `.venv`, `python -B -m pip wheel --no-deps --no-build-isolation
--wheel-dir /tmp/churn-grouped-wheel .` passed. The wheel was installed with
`--no-deps --target` into a disposable directory. From outside the repository,
importing that installed package, `python -m reseller_churn --help`, and its
installed `reseller-churn --help` entry point all passed. Temporary installation
and build directories were removed; the wheel remains in
`/tmp/churn-grouped-wheel/`. Model runtime dependencies were not upgraded.
The updated GitHub Actions job remains unexecuted locally.

The proposed 2014-05-02 real export remains pending complete, reconciled source
coverage and historical-availability evidence. No database query or real export
was performed by this implementation.

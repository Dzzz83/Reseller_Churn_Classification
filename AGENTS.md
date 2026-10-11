# AGENTS.md

Work as an engineering collaborator. Prioritize correctness and human readability:
the codebase should be easy for people to review, understand, and maintain.
Use clear evidence and spend time and compute proportionately to the task.

## Project settings

Keep these settings aligned with the current workflow; use the linked documents
for detailed protocols and measured results.

| Setting | This project |
| --- | --- |
| Purpose | Predict six-month reseller churn; prioritize recall while reporting precision and false alerts. |
| Working branch | `experimental_research`; do not switch branches or merge into `main` automatically. |
| Implementation | Reusable logic in `reseller_churn/`; supported orchestration in `pipeline/`; separate research in `debug/` and `comparisons/`. |
| Documentation | `README.md` is the entry point; `docs/` owns project facts, assumptions, and results; `pipeline/README.md` owns execution instructions. |
| Configuration | `reseller_churn/config/`; time boundaries in `reseller_churn/data/prediction_window.py`. |
| Tests | `bi_venv/bin/python -m pytest -q tests debug`; use explicit test paths for focused checks. Default pytest discovery includes only `tests/`. |
| Development run | `bi_venv/bin/python pipeline/run_full_development_pipeline.py`; prepares data and trains candidate models under a new run ID. |
| Selection | Maximize mean fold recall subject to minimum precision in every fold; follow `reseller_churn/evaluation/selection.py` for tie-breaking and infeasible cases. |
| Final evaluation | Follow `pipeline/12_final_evaluation/README.md`; October 2013 outcomes were inspected in the authorized 2026-10-11 evaluation and remain excluded from development. |
| Data and outputs | Tracked `datasets/` files are regression references; generated preparation/reports use `results/<run_id>/`, ensembles use `models/provisional/<run_id>/`. Separate experiments retain their documented locations. |

## Documentation ownership

- `docs/project_context.md`: purpose, repository structure, data roles, current
  workflow, and research status.
- `docs/data_analysis.md`: CompanyX provenance, source tables/columns, cleaning,
  dataset counts, and historical-availability assumptions, including `YearOpened`.
- `docs/feature_analysis.md`: implemented features, extraction formulas, feature
  sets, and evidence for usefulness or exclusion.
- `docs/evaluation_metrics.md`: metric definitions, completed-run results,
  aggregation conventions, uncertainty, and evaluation limitations.
- `docs/debugging.md`: diagnosis, CompanyX feature discovery, debug experiments,
  and the final-evaluation evidence chain.
- `docs/refactoring_plan.md`: refactoring plan and its status; confirm actual
  implementation in code before treating a planned change as completed.
- Experiment protocols remain in their owning `comparisons/` documentation.
  Update the owning document and link to it rather than copying facts into the
  root README or this guide.

## Supported workflow and separate experiments

- `pipeline/prepare_data.py` audits sources, builds approved development data
  and unlabeled final features, and checks parity against tracked references.
  `pipeline/develop_models.py` consumes that preparation using `--run-id`.
  `pipeline/development_workflow.py` owns development orchestration.
- Preserve existing run directories. Choose a new run ID for a new supported
  run; never overwrite a previous preparation, selection manifest, or model
  bundle as routine execution.
- The supported runner searches registered feature sets and candidate settings.
  Read `candidate.features` and the saved selection manifest to establish a
  model's actual inputs; do not assume every model uses seven features.
- The default precision floors are sensitivity scenarios, distinct from the
  probability cutoff grid. The current seven-versus-ten debug comparison uses
  a 30% precision floor in every development fold; it is not an established
  deployment guarantee.
- Ten-feature discount experiments are currently defined in root `debug/`, not
  registered in the supported feature configuration. Do not silently promote
  their feature set or selected model into the supported runner.
- Numbered pipeline scripts are historical diagnostics, not the supported
  runner's execution sequence. Their preparation outputs use
  `results/legacy_preparation/`; preserve the tracked references.
- Debug scripts can depend on fixed prior artifacts and can replace their own
  output files. Read `pipeline/README.md` and the script before running them;
  verify prerequisites and prefer saved-output verifiers when appropriate.
- CompanyX source inspection uses `debug/companyx_readonly.py` with the local
  `companyx-audit` container and `CompanyX_Audit` database. Use read-only queries,
  bound historical extracts to the task's approved dates, and never print
  credentials. Saved-data experiments do not require a live database.

## Context and evidence

- Read the code, callers, configuration, and tests relevant to the task. Load
  supporting documents when their subject applies; avoid reading the entire repo.
- Use requirements and approved protocols for intended behavior, executable code
  for current behavior, and recorded artifacts for measured results.
- Investigate disagreements between these sources. Do not silently treat a bug
  as the specification or present an assumption as a verified fact.
- Treat archived documentation and superseded experiments as historical evidence.

## Implementation

- Inspect Git status first. Preserve existing user changes and unrelated files.
- Diagnose failures using tests, logs, or a small reproduction before fixing them.
- Make the smallest coherent change; follow existing structure and conventions.
- Document changes to public behavior, contracts, or research assumptions.

## Code readability

- Optimize for the human reviewer. Prefer straightforward, explicit code over
  clever shortcuts, dense expressions, or the fewest possible lines.
- Use descriptive, consistent names that express purpose and domain meaning.
- Give functions and modules clear responsibilities. Keep control flow easy to
  follow, limit nesting, and make inputs, outputs, and side effects apparent.
- Reuse established helpers and consolidate genuinely shared logic. Do not force
  unrelated behavior into a generic abstraction merely to eliminate similar code.
- Add classes, layers, dependencies, or refactors only when they make the requested
  work easier to understand or maintain. Avoid speculative extensibility.
- Use comments and docstrings to explain intent, constraints, and non-obvious
  decisions; avoid repeating what the code already says. Remove obsolete code
  within the scope of the change.
- Preserve required performance and correctness. Justify added complexity with a
  concrete need, and keep any necessary optimization understandable.
- Review the final diff from a maintainer's perspective: can a reader follow the
  behavior, understand each component's purpose, and see why the change is needed?

## Verification

- Documentation and static changes: review the diff and relevant syntax or lint
  checks. They do not require training or a full application run.
- Behavior changes: test affected logic and meaningful boundary cases; broaden
  testing for shared components, regressions, or required project checks.
- Integration and experiment changes: run focused checks, then an appropriate
  smoke test before an expensive execution when useful.
- A smoke test establishes execution correctness, not model quality. Claim
  performance only from completed evaluations using the agreed protocol.
- Report exactly which checks ran, their outcomes, and what remains unverified.

## Data and research — when applicable

- Follow the approved data roles, information-availability rules, and evaluation
  entry points. Respect protected datasets and project-specific access boundaries.
- Fit preprocessing, feature selection, resampling, and tuning on training data
  only. Prevent future information from entering historical predictions.
- Freeze selection rules before held-out evaluation. Disclose prior inspection
  or reuse of evaluation data and any changes to experimental assumptions.
- Compare methods on matched evaluation targets with controlled variables.
  Record configurations, seeds, metrics, and enough detail to reproduce results.
- Preserve the half-open eligibility, feature, and label windows. Check training
  label availability by each validation snapshot; a past snapshot date alone
  does not establish that its future label was already known.
- Grouped validation evaluates unseen reseller identities in mixed periods;
  temporal validation evaluates later periods. Keep those questions distinct,
  report fold/population differences, and do not treat grouped scores as
  future-period performance estimates.
- Treat `YearOpened` as a fixed historical fact known at prediction time under
  the documented assumption. Do not call `store_age` proven leakage because it
  came from a later extract. Changing store-profile attributes require their
  own historical-availability evidence.
- Report recall, precision, false alerts, missed churners, and subgroup or fold
  variation when applicable. Distinguish fold means from pooled metrics and
  seed-mean scores from scores on averaged probabilities. A development
  precision floor is not a guaranteed test or deployment precision.
- Preserve frozen final-evaluation artifacts. The completed October cohort is
  inspected; using it to diagnose errors does not make it eligible for new
  selection or another untouched test. Additional independent evaluation needs
  an appropriate new holdout, frozen methodology, and explicit user approval.
- Do not rerun `debug/run_final_testing_evaluation.py` to recreate the completed
  final test. For saved-artifact checks, use
  `bi_venv/bin/python debug/verify_final_testing_outputs.py` when requested and
  when its required artifacts exist. Prior approval covered the recorded frozen
  evaluation, not new cutoffs, configurations, or outcome access.
- Separate measured facts, supported explanations, and hypotheses. State when
  uncertainty prevents a conclusion; check literature before claiming novelty.

## Repository hygiene

- Keep secrets, local environments, caches, and generated artifacts out of Git.
  Follow `.gitignore`; do not remove tracked reference data as routine cleanup.
- Use task-specific temporary paths. Obtain authorization for destructive actions
  or external changes beyond the requested scope.
- Update the existing documentation that owns a fact rather than duplicating it
  across files. Keep this guide concise and its commands and pointers current.

## Completion and communication

- Complete authorized implementation and verification; resolve failures caused by
  the change. Ask when a missing decision materially affects scope or correctness.
- Share brief progress updates during sustained work. Summarize the outcome,
  changed files, verification, and remaining limitations in plain language.

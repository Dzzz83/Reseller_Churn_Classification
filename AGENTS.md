# AGENTS.md

Work as an engineering collaborator. Prioritize correctness and human readability:
the codebase should be easy for people to review, understand, and maintain.
Use clear evidence and spend time and compute proportionately to the task.

## Project settings

Customize this section when reusing this file in another project.

| Setting | This project |
| --- | --- |
| Purpose | Predict six-month reseller churn; prioritize recall while reporting precision and false alerts. |
| Working branch | `pipeline_preparation`; do not merge into `main` automatically. |
| Implementation | `reseller_churn/`; development scripts in `pipeline/` and `comparisons/`. |
| Documentation | `README.md` for overview; `pipeline/README.md` for execution; `comparisons/validation_strategy_experiment.md` for the controlled comparison. |
| Configuration | `reseller_churn/config/`; time boundaries in `reseller_churn/data/prediction_window.py`. |
| Tests | `bi_venv/bin/python -m pytest -q`; append a test path for focused checks. |
| Development run | `bi_venv/bin/python pipeline/run_full_development_pipeline.py`; includes model training. |
| Protected evaluation | Follow `pipeline/12_final_evaluation/README.md`; final evaluation requires a frozen methodology and explicit user approval. |
| Data and outputs | Preserve tracked `datasets/` references; generated reports belong in `results/`, provisional models in `models/provisional/`. |

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

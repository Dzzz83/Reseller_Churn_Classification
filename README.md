# Reseller Churn Classification

Predict whether a reseller active in the preceding six months will place no
orders during the following six months. The project is in development; model
and threshold choices remain provisional.

## Documentation

- [2026-10-16 machine handoff](handoff/2026-10-16/README.md): restore the saved
  research/data/presentation archives and continue from another checkout.
- [Project context](docs/project_context.md): purpose, repository structure,
  prediction contract, model settings, validation, and research status.
- [Code architecture](docs/code_architecture.md): module responsibilities, fixed
  scoring versus research workflows, and saved-model compatibility.
- [Baseline error diagnosis](comparisons/baseline_error_diagnosis.md): saved
  development errors, ranking overlap, behavioral slices, and recurrence checks.
- [Research journal](docs/research_journal.md): current continuation work,
  verification, completed cycles, and links to their frozen protocols.
- [Data analysis](docs/data_analysis.md): CompanyX backup provenance, source
  tables and columns, row counts, CSV mappings, cleaning, and availability assumptions.
- [Feature analysis](docs/feature_analysis.md): candidate feature sets,
  extraction formulas, alternative features, and measured development evidence.
- [Evaluation metrics](docs/evaluation_metrics.md): metric explanations, latest
  completed-run results, false alerts, and evaluation limitations.
- [Lessons and research rules](docs/research_rules.md): what the experiments
  established, the failed generalization, and requirements for future comparisons.
- [Debugging and feature experiments](docs/debugging.md): missed-churn diagnosis,
  CompanyX feature discovery, grouped precision improvement, and component ablations.
- [Repeated grouped precision study](comparisons/grouped_precision_recall.md):
  nested feature/parameter selection with recall above 90%.
- [Component ablation study](comparisons/grouped_ablation.md): contributions of
  cadence features, tree size, and L2, including adjusted uncertainty.
- [Pipeline execution](pipeline/README.md): supported preparation/development
  workflow, historical diagnostics, and optional commands.
- [Controlled validation-strategy experiment](comparisons/validation_strategy_experiment.md):
  the matched-population retrospective comparison protocol.
- [Alternative-method audit](comparisons/correlation_pruned_churn/README.md):
  independent reconstruction and validation protocols.

The [development research preparation protocol](comparisons/development_research_protocol.md)
defines development-only checks,complete period/subgroup comparisons,cutoff-transfer
diagnostics and the requirements to freeze before another candidate. Preparation
runs no new model experiment or final test.

## Current development findings

**Current conclusion:** no replacement has met the complete current development
acceptance criteria. Source, feature, learner and score-transfer follow-ups are
complete. The latest paid-unit-revenue candidate meets point recall/false-alert
requirements but fails the frozen false-alert uncertainty requirement. See
[current evidence](docs/evaluation_metrics.md#latest-fixed-numeric-feature-result--2026-10-17)
and the [continuation checkpoint](docs/continuation_checkpoint.md).
The explicitly authorized October test also fails: the latest candidate catches
45/46 with194 false alerts, versus43/46 with108 for historical0.06 and46/46 with183
for the baseline using the same development cutoff objective. See
[October test results](docs/evaluation_metrics.md#authorized-latest-candidate-october-test--2026-10-17).

The earlier cadence candidate improved grouped development validation
but failed to improve October precision. The gains below are development
evidence, not a demonstrated improvement on new populations or a deployment
recommendation. Follow the [research rules](docs/research_rules.md) for further work.

The completed studies below maximize precision while maintaining **grouped recall
above 90%**, using development data only. The current objective is stricter:
**fewer false alerts while matching or exceeding baseline recall on the same
development splits**. See the [research journal](docs/research_journal.md) for
the ongoing matched-history investigation and predeclared follow-up.
Three repetitions of five outer reseller
folds use three inner grouped folds for feature, parameter, and cutoff selection.
All snapshots of a reseller stay together; outer labels never tune cutoffs.

| Development comparison | Mean recall | Mean precision |
| --- | ---: | ---: |
| Original ten features, training-selected cutoff | 91.51% | 27.16% |
| Adaptive inner feature/parameter selection | 91.51% | 29.74% |
| Fixed cadence/regularization configuration | 94.65% | 29.31% |
| Original ten plus purchase-gap variation only | 92.14% | 28.89% |

The adaptive procedure improves precision by **2.59 percentage points** at
unchanged mean recall, with about 62 fewer false alerts per repetition. Every
repetition exceeds 90% recall, although the weakest is 90.09% and four of 15
individual folds fall below 90%. The fixed-configuration rows describe separate
ablations; they are not alternative names for the adaptive procedure's result.

Fourteen controlled ablations identify **purchase-gap variation and overdue
ratio** as the clearest precision contributors. Seven leaves mainly help
recall; annual order count and L2 precision effects remain uncertain after
adjusting across 17 comparisons. The current research candidate keeps the
original ten features plus three cadence features, seven leaves, L2=1, and a
development-selected cutoff. The subsequent October follow-up below finds
lower precision at that frozen cutoff; it is not installed into the operational
pipeline.

The adaptive and fixed-configuration rows use the same development dataset,
not two independent validation sets. They evaluate different objects: a
configuration-selection process versus one configuration across outer refits.
Neither directly evaluates the single full-development fitted bundle. The
development baseline uses high-recall training-selected cutoffs, whereas the
October historical baseline uses 0.06; the differences do not isolate the
feature/parameter effect under a common operating policy.

See the [latest metrics](docs/evaluation_metrics.md#grouped-precision-and-ablation-results--2026-10-16),
[grouped study](comparisons/grouped_precision_recall.md), and
[ablation findings](comparisons/grouped_ablation.md#completed-results--2026-10-16)
for exact protocols, repeat results, uncertainty, and limitations. These are
exploratory development findings; no new independent final performance is claimed.

### Earlier development comparisons

The earlier seven-versus-ten experiment compares seven purchasing features against
those seven plus three discount features across Logistic Regression, Random
Forest, and histogram gradient boosting. It uses temporal validation and five
stratified folds grouped by reseller, maximizing mean recall with **precision
at least 30% in every fold**.

Seven-feature Random Forest has the highest temporal mean recall, **36.59%**.
Ten-feature gradient boosting reaches **33.81% temporal recall** and **75.51%
grouped recall**; these methods evaluate different populations and questions.
Discounts improve boosting, preserve Logistic Regression's temporal recall
with fewer false alerts, and reduce the forest's mean recall. No universal
ten-feature replacement or deployment model has been selected.

## Frozen candidate October follow-up

The user authorized evaluating the saved candidate after development and
ablations. At its unchanged cutoff it catches 45 of 46 churners with 179 false
alerts: **97.83% recall and 20.09% precision**, versus the original rule's
93.48% / 28.48% with 108 false alerts. It catches two more churners but adds
71 net false alerts, so **it fails the precision-improvement objective on
October**. Overall AP improves, but active-subgroup AP worsens.

This is descriptive reuse of already inspected October outcomes. No model is
refitted, no cutoff is changed, and no alternative is selected from these
results. See [the evaluation record](comparisons/frozen_october_evaluation.md).

The [failure diagnosis](comparisons/frozen_failure_diagnosis.md) explains the
validation-to-October precision drop using saved predictions, audits the
different baseline cutoff policies, and identifies development subgroup
instability. It changes no model or cutoff.

Candidate active false-positive rates are about 38% in both validation and
October, while active churn rates fall from 9.91% to 2.46%. Development subgroup
operating rates applied to October counts imply 20.17% precision, close to
20.09% observed. This supports a population explanation without proving one
cause. Both models face the same October accounts, and the original rule still
produces fewer false alerts. Future comparisons must use matched cutoff policies,
report subgroup weaknesses and prevalence sensitivity, and preserve untouched
data for independent evaluation; see [the full rules](docs/research_rules.md).

<a id="authorized-final-testing"></a>

## Original authorized final testing

On 2026-10-11, the user froze ten-feature gradient boosting with the grouped
30% precision rule as primary and three predeclared secondary rules. The
October 2013 test has 492 resellers and 46 churners. The primary rule achieves
**93.48% recall and 28.48% precision**, with 43 churners caught, three missed,
and 108 false alerts. **The intended 30% precision level is not met on this test.**
The saved predictions preceded outcome access, and no cutoffs were changed
afterward. These test outcomes are now inspected; the model remains provisional.
See [final metrics](docs/evaluation_metrics.md#final-test-evaluation--2026-10-11).

Recently active accounts account for eight caught churners and 98 false alarms;
the high overall recall therefore needs subgroup context. The
[debugging report](docs/debugging.md#development-precision-diagnosis--2026-10-16)
records the subsequent development diagnosis and need for an untouched cohort
for an independent final claim. The grouped improvement and ablation studies
do not access final testing.

Earlier development metrics and tradeoffs are in [evaluation metrics](docs/evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features).
The historical ten-feature comparison runs through root `debug/`; the supported pipeline
still registers RFM, Pruned, Reduced, and Full. See the
[debug execution guide](pipeline/README.md#debugging-and-feature-experiments).

## Set up and run

The standalone grouped studies have [reproduction commands](comparisons/grouped_ablation.md#execution-and-verification).
The module/package import collisions are resolved: frozen settings live in
`config/frozen_model.py`, strict preparation in `data/preparation.py`, and
explicit cutoff reports in `evaluation/cutoff_evaluation.py`. Both workflows
use shared window, label, metric, and artifact helpers. See
[code architecture](docs/code_architecture.md) for ownership and compatibility.

Use Python 3.11 or later in a project virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python pipeline/run_full_development_pipeline.py
```

For fixed ten-feature scoring, start with `python -m reseller_churn --help`
or `reseller-churn --help`. These expose `prepare`, `features`, `train`,
`predict`, and `evaluate`; they do not search models or select cutoffs.
Required source inputs are documented in the [input contract](docs/input_contract.md).

The runner prepares data and compares model configurations on temporal folds,
prioritizing recall at several minimum precision levels. See the
[pipeline guide](pipeline/README.md) for options and output files. Final-test
evaluation requires a frozen methodology and explicit user approval; follow
[Stage 12 instructions](pipeline/12_final_evaluation/README.md).

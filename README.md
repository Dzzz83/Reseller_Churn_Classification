# Reseller Churn Classification

Predict whether a reseller active in the preceding six months will place no
orders during the following six months. The project is in development; model
and threshold choices remain provisional.

## Documentation

- [Project context](docs/project_context.md): purpose, repository structure,
  prediction contract, model settings, validation, and research status.
- [Data analysis](docs/data_analysis.md): CompanyX backup provenance, source
  tables and columns, row counts, CSV mappings, cleaning, and availability assumptions.
- [Feature analysis](docs/feature_analysis.md): candidate feature sets,
  extraction formulas, alternative features, and measured development evidence.
- [Evaluation metrics](docs/evaluation_metrics.md): metric explanations, latest
  completed-run results, false alerts, and evaluation limitations.
- [Debugging and feature experiments](docs/debugging.md): missed-churn diagnosis,
  CompanyX feature discovery, feature comparisons, and the nested grouped revenue study.
- [Pipeline execution](pipeline/README.md): supported preparation/development
  workflow, historical diagnostics, and optional commands.
- [Controlled validation-strategy experiment](comparisons/validation_strategy_experiment.md):
  the matched-population retrospective comparison protocol.
- [Alternative-method audit](comparisons/correlation_pruned_churn/README.md):
  independent reconstruction and validation protocols.

## Current development findings

The latest completed study compares SubTotal and TotalDue purchasing amounts
for seven-feature Random Forest and ten-feature gradient boosting using repeated
nested grouped validation. It found **no reliable TotalDue improvement** under
the frozen recall/precision criterion. No revenue-feature replacement or new
deployment cutoff was selected. See the
[nested results and threshold reports](docs/evaluation_metrics.md#nested-grouped-revenue-comparison--2026-10-15).

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

## Authorized final testing

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
[debugging report](docs/debugging.md#recommended-follow-up--not-yet-executed)
records the proposed follow-up diagnosis and need for a new independent holdout.

Subsequent RF7 and revenue-variant evaluations reuse these inspected October
outcomes. TotalDue boosting reduced false alerts on that cohort, but the nested
development study did not establish reliable improvement. These retrospective
comparisons do not constitute new independent tests. See the
[October revenue comparison](docs/evaluation_metrics.md#october-revenue-comparison-on-inspected-outcomes--2026-10-15).

Development metrics and tradeoffs are in [evaluation metrics](docs/evaluation_metrics.md#seven-versus-ten-feature-comparison).
The ten-feature comparison runs through root `debug/`; the supported pipeline
still registers RFM, Pruned, Reduced, and Full. See the
[debug execution guide](pipeline/README.md#debugging-and-feature-experiments).

## Set up and run

Use Python 3.11 or later in a project virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python pipeline/run_full_development_pipeline.py
```

The runner prepares data and compares model configurations on temporal folds,
prioritizing recall at several minimum precision levels. See the
[pipeline guide](pipeline/README.md) for options and output files. Final-test
evaluation requires a frozen methodology and explicit user approval; follow
[Stage 12 instructions](pipeline/12_final_evaluation/README.md).

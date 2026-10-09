# Reseller Churn Classification

Predict whether a reseller active in the preceding six months will place **no
orders during the following six months**. Predictions use historical data only.

The repository separates reusable implementation from numbered experiment
scripts. The project is currently in **development**: model and threshold choices
are provisional; the protected final test has not been evaluated.

## Structure

```text
reseller_churn/
  config/        file paths, feature sets, model and fold settings
  data/          prediction windows, eligibility, churn labels and folds
  features/      historical feature construction
  modeling/      model factories, oversampling and seeded predictions
  evaluation/    shared metrics and regression checks
pipeline/        numbered development stages (01–12)
comparisons/
  correlation_pruned_churn/    independent method and validation audit
tests/
  step_01_data/
  step_02_features/
  step_03_modeling/
  step_04_evaluation/
```

See [pipeline/README.md](pipeline/README.md) for stage-by-stage execution.

## Set up and run

Use Python 3.11 or later in a project virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python pipeline/run_full_development_pipeline.py
```

The development runner executes stages 01–08 and 10–11, including development
model evaluation. Stage 09 exhaustive tuning is optional and does not run by
default. The runner **never executes Stage 12**.

Additional, separate development-only comparisons:

```bash
python pipeline/06_model_selection/02_compare_model_design_factors.py
python comparisons/correlation_pruned_churn/03_compare_validation_strategies.py
```

The correlation-pruned audit also provides validation, feature-verification,
comparison, and original-method reproduction scripts documented in its README.

## Prediction and leakage boundaries

At snapshot `t`:

- Eligibility: at least one order in **[t − 6 months, t)**.
- Features: orders strictly before `t` (up to 12 months of history in
  the main feature pipeline).
- Churn label: **1** when no orders occur in **[t, t + 6 months)**;
  otherwise **0**. The upper boundary is exclusive.
- Preprocessing, supervised feature selection, oversampling, and model
  training are fitted only on training rows within each fold.
- The **2013-10-01** final-test snapshot is protected; its outcome labels
  are not available to development code. The stored final-test CSV contains
  features, not churn labels.

## Development validation designs

The main development pipeline uses chronological folds:

| Fold | Training snapshots | Validation snapshot |
| --- | --- | --- |
| 1 | 2012-07-01 | 2013-01-01 |
| 2 | 2012-07-01, 2012-10-01 | 2013-04-01 |

The comparison module also supports **GroupKFold by StoreID**, which measures
performance on unseen reseller identities within mixed historical periods.
Chronological validation instead asks how a model performs on later periods.
Both are legitimate for their respective questions, but results from different
snapshot schedules and populations must not be interpreted as a direct
head-to-head comparison.

Historically unverified store-profile attributes are isolated in an explicitly
opt-in diagnostic comparison; they are not silently treated as past-known data.

## Metrics and research decisions

The current locked feature baseline contains:

```text
n_orders_3m, revenue_3m, recency_days,
share_bikes, share_accessories, share_clothing, revenue_12m
```

Model settings, random seeds, and provisional thresholds are defined in
`reseller_churn/config/model_settings.py`. Ranking metrics such as PR-AUC are
distinct from precision/recall/F1 calculated at a specified threshold. The
existing threshold script ranks development thresholds by **mean F1**; choosing
a new recall-first objective is a **future model-selection experiment**, not an
implicit change made by this refactor. The business goal makes churn recall
important, but precision, false alarms and stability must also be reported.

## Reproducibility and generated files

- `datasets/orders_clean.csv` and `datasets/stores_clean.csv` are cleaned
  input datasets.
- The three tracked `datasets/processed/` CSVs are verified references for
  feature and ML dataset assembly. Do not silently replace them.
- `results/`, `models/provisional/` and `*.log` are **generated outputs**,
  excluded from future Git commits. Regenerate them by running the appropriate
  stage. Old experiment results are not retained as a repository archive.
- `python -m pytest -q` verifies protected boundaries, feature/dataset
  parity, model baselines, shared metrics and validation safeguards.
- GitHub Actions runs both regression/audit checks and a separate full
  development-pipeline verification on changes to the pipeline.

Refactoring must preserve observed data and previously locked model scores.
Changes to feature definitions, validation policy, thresholds or model choice
must be evaluated and documented as separate research decisions.

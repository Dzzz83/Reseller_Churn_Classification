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
# Reseller churn: final pipeline

Predict six-month reseller churn using the fixed ten-feature histogram gradient
boosting model. Save ranked scores once, then compare supplied probability
cutoffs without retraining. This branch contains the final workflow; model
searches and experiment history remain on separate research branches. The
original model source is `pipeline_preparation` at commit `f53202d`; later
revenue comparisons and SHAP analysis are preserved on
`research/revenue-shap-results-20261016`.

## Latest research findings

The repeated nested grouped comparison did not establish a reliable advantage
from replacing SubTotal with TotalDue. SHAP explains how held-out predictions
use the current features, including a high-score false alert; it does not prove
causation or prediction correctness. These results do not change the fixed
model or its original cutoff. See [results, thresholds, and limitations](docs/model_and_results.md#later-research-findings).

## Install and verify

Use Python 3.11 or later in a virtual environment:

```bash
python -m pip install ".[dev]"
python -m pytest -q
python pipeline/smoke_test.py
reseller-churn --help
```

The smoke uses synthetic inputs and outcomes. It checks execution, not model
quality. From a source checkout, `python pipeline/run_pipeline.py` or
`python -m reseller_churn` accepts the same subcommands as `reseller-churn`.

## Input files

Provide four CSVs. [The input contract](docs/input_contract.md) defines their
columns, validation, coverage declarations, and feature formulas.

| Input | Purpose |
| --- | --- |
| Clean orders | Accepted order IDs, dates, quantities, subtotals, and category revenue |
| Stores | Valid reseller IDs |
| Order headers | Source order-to-reseller mapping, dates, and subtotals |
| Order lines | Quantities, prices, discounts, and line totals required by discount features |

The tracked `datasets/orders_clean.csv` and `datasets/stores_clean.csv` supply
the first two inputs. They do **not** supply the order-line discount inputs.
Export those separately from the source restoration; the optional
[SQL template](pipeline/export_order_inputs.sql) produces the two result sets.
The pipeline does not need Docker or a database connection once CSVs exist.
Store local exports under ignored `inputs/`.

`--coverage-from` and `--coverage-through` declare complete source coverage,
including dates with no purchases. The latter is an exclusive endpoint.
Do not infer completeness from the last observed order date.

## Prepare the approved development rows

```bash
reseller-churn prepare \
  --orders datasets/orders_clean.csv --stores datasets/stores_clean.csv \
  --headers inputs/order_headers.csv --lines inputs/order_lines.csv \
  --coverage-from 2011-05-31 --coverage-through 2013-10-01 \
  --output results/run-001/preparation
```

This builds the ten features and six-month labels for July/October 2012 and
January/April 2013. It never constructs October 2013 final outcomes.
Declared coverage must contain every required history and development-label
window. No search, tuning, or threshold optimization is performed.

## Train the fixed ensemble

```bash
reseller-churn train \
  --development results/run-001/preparation/development.csv \
  --prediction-date 2013-10-01 --output models/run-001
```

Training uses balanced weights from development labels and averages the five
registered seeds. All training labels must resolve by the declared prediction
date. [The model specification](docs/model_and_results.md) records the fixed
parameters and original evaluation limitations.

The bundle includes checksums, configuration, input hashes, package code
hashes, and runtime versions. Loading rejects incompatible or modified bundles.
Use only trusted model files: joblib deserialization can execute code.

## Build scoring features and save the ranked probabilities

```bash
reseller-churn features \
  --orders datasets/orders_clean.csv --stores datasets/stores_clean.csv \
  --headers inputs/order_headers.csv --lines inputs/order_lines.csv \
  --coverage-from 2011-05-31 --coverage-through 2013-10-01 \
  --snapshot 2013-10-01 --output results/run-001/features

reseller-churn predict \
  --model models/run-001 --features results/run-001/features/features.csv \
  --output results/run-001/prediction
```

`ranked_predictions.csv` contains `StoreID`, `snapshot`, `n_orders_3m`,
`churn_probability`, and `rank`. Accounts are ranked by descending score, with
StoreID breaking ties; rank restarts per snapshot. Scoring never reads outcomes.
Scores are model probability outputs, not validated calibrated individual risks.
Later snapshots need correspondingly complete source exports. The training
snapshot set remains fixed; changing it is a methodology change.

## Compare your score cutoffs

Provide observed outcomes as a separate CSV with `StoreID`, `snapshot`, and
`actual_churn` (0 or 1). Every scored account must have exactly one matching
outcome. A churn label means no accepted purchase during `[snapshot,
snapshot + 6 months)`. Outcomes must cover the full six months.

```bash
reseller-churn evaluate \
  --predictions results/run-001/prediction/ranked_predictions.csv \
  --outcomes inputs/observed_outcomes.csv \
  --cutoffs 0.06 0.10 0.30 0.55 \
  --outcomes-through 2014-04-01 --cohort-status inspected \
  --output results/run-001/evaluation
```

An account is alerted when its score is **greater than or equal to** the cutoff.
Changing cutoffs never refits the model. The command prints an overall table and
saves:

- `cutoff_metrics.csv`: recall, precision, F1, TP/FP/FN/TN, alert counts/rates,
  average precision, and ROC-AUC, per snapshot and recent-activity subgroup.
- `cutoff_decisions.csv`: account-level decisions and error categories at each cutoff.
- `manifest.json`: supplied cutoffs, input hashes, declared cohort status and
  coverage, output hashes, and execution identity.

Original October cutoffs retain their primary/secondary roles; additional
cutoffs are marked `additional`. Different snapshots are reported separately.
Undefined single-class ranking metrics are blank, and zero-denominator
threshold precision/recall/F1 are reported as zero.

October 2013 outcomes have already been inspected. All further cutoff comparisons
on that cohort are descriptive reuse. The original primary cutoff is 0.06;
evaluating alternatives does not retrospectively replace it or establish new
held-out performance. A future independent evaluation needs externally frozen
methodology and cutoffs, verified prediction-before-outcome chronology, and
explicit approval. `--cohort-status independent` requires
`--approve-independent-evaluation` and rejects the known October cohort.
The CLI records that declaration; it cannot independently establish historical
chronology or source completeness. It never extracts new protected outcomes.

## Output and repository boundaries

Use a new output directory for every stage/run; existing directories are
rejected. Generated `results/`, `models/`, local exports, and environments remain
ignored. Preserve tracked datasets as read-only source references.

`reseller_churn/` contains the final implementation; `pipeline/` contains the
source entry point, optional extraction SQL, and synthetic smoke. Tests and CI
cover this workflow without research outputs or the audit database. Original
experimental scripts, local evaluation artifacts, and hashes are preserved in
the research checkout. See [verification and review](docs/verification.md).

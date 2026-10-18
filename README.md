# Reseller churn: production pipeline

Predict six-month reseller churn with the final ten-feature period/class-weighted
histogram boosting ensemble. The `export` command loads an existing model and
writes one Power BI snapshot without fitting, selecting a cutoff, or accessing
outcomes. Research code and results remain on research branches.

## Install and verify

Use Python **3.14.8** in a virtual environment. Runtime dependencies are pinned
in `pyproject.toml` to the retained model's environment. Loading also checks the
Python patch version; do not upgrade dependencies or retrain to bypass a mismatch.

```bash
python -m pip install ".[dev]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python pipeline/smoke_test.py
python -m reseller_churn --help
```

`reseller-churn` is the installed entry point; `python pipeline/run_pipeline.py`
provides the same commands from a checkout. The synthetic smoke exercises
nested grouped development, five-seed fitting, artifact loading, and CSV export.
It verifies execution, not model quality. Package code has no research imports.

## Input files

Supply explicit CSV paths for orders, stores, headers and lines, plus a
hash-bound completeness manifest. [The input contract](docs/input_contract.md)
defines schemas, reconciliation, coverage, exclusions and feature formulas.

Tracked `datasets/orders_clean.csv` and `datasets/stores_clean.csv` remain source
references. They do not supply header/detail facts or establish complete source
coverage. Obtain reconciled local CSV exports independently and keep them under
ignored `inputs/`. The optional `pipeline/export_order_inputs.sql` is an extraction template for
separate source preparation, not an automatic database step. The pipeline needs
no database connection. Coverage must
include dates with no purchases; a maximum observed transaction date is
insufficient evidence of completeness.

## Load the retained artifact

Keep the existing `frozen_models.joblib` unchanged outside Git, or copy it locally
to `models/period_class_weighted/frozen_models.joblib`. Supply its path explicitly;
there is no hosted binary download. The packaged
`reseller_churn/config/retained_model.json` supplies its checksum, runtime,
feature/model contract and exact cutoff. It requires no adjacent research
manifest, scores or performance files. Only load trusted joblib files.

```bash
reseller-churn validate-model \
  --model models/period_class_weighted/frozen_models.joblib
```

This command validates all five fitted estimators without fitting or predicting.
See [model and cutoff specification](docs/model_and_results.md).

## Export a Power BI snapshot

After the source owner has reconciled and attested complete inputs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 reseller-churn export \
  --model models/period_class_weighted/frozen_models.joblib \
  --orders inputs/orders.csv --stores inputs/stores.csv \
  --headers inputs/headers.csv --lines inputs/lines.csv \
  --source-manifest inputs/source_coverage.json \
  --coverage-from 2013-05-02 --coverage-through 2014-05-02 \
  --snapshot 2014-05-02 --output results/power-bi-20140502
```

This is a command example, **not an export already performed**. For this snapshot,
history is `[2013-05-02, 2014-05-02)` and eligibility is
`[2013-11-02, 2014-05-02)`. The pipeline derives the eligible count from purchases;
the pipeline does not hard-code an expected population. A separately authorized
source check on 2026-10-18 verified 467 eligible resellers against the supplied
CompanyX backup. Complete source extracts and a production coverage manifest
were saved locally; see [source readiness](docs/verification.md#remaining-export-readiness).
The earlier exports ending at 2013-10-01 are insufficient. Use the newly verified
full-window inputs for this snapshot. Implementation/model checks used synthetic
data; no real prediction export has been run.

The new output directory contains:

- `predictions.csv`: one row per eligible reseller, sorted by StoreID, with
  `StoreID,Snapshot`, all ten estimator features in order, `Model score` and
  `Predicted churn`. Snapshot uses `YYYY-MM-DD`; flags are `Yes`/`No`.
- `manifest.json`: exact cutoff, model identity/runtime/checksum, mean probability
  aggregation, source hashes/declarations, effective and loaded coverage,
  exclusions, row count and output hash, `models_fitted=0`, `outcomes_accessed=false`.

Features and scores retain round-trip numeric precision. Missing discount values
are blank; there is no imputation or display rounding. Flags use full-precision
`score >= 0.27912545951329487`. Power BI should import Snapshot as Date, StoreID
as whole number, features/scores as numeric and Predicted churn as text.
Scores are model outputs, not established calibrated individual risks.

## Develop a future model update

The immediate export uses the retained artifact and cutoff. A separate command
supports future updates with the same model family and four approved development
snapshots. It uses fitting-partition period/class weights and nested reseller
groups; it never reads or builds protected final-test outcomes.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 reseller-churn develop \
  --orders inputs/orders.csv --stores inputs/stores.csv \
  --headers inputs/headers.csv --lines inputs/lines.csv \
  --source-manifest inputs/source_coverage.json \
  --coverage-from 2011-05-31 --coverage-through 2013-10-01 \
  --output models/development-update-001
```

This intentionally trains and selects a development cutoff; it must not be used
as an artifact import or as preparation for the immediate export. Outputs include
inner OOF scores, outer grouped decisions, repeat/snapshot recall, precision and
false alerts, and a new model/manifest when the precision-floor rule is feasible.
A failed selection records `no_feasible_cutoff` and saves no model. For an
explicitly approved future model, pass `--model-manifest` with that artifact's
`model_manifest.json` to `validate-model` or `export`.

## Output and repository boundaries

Use a fresh output directory for each run; existing directories are refused.
Generated `results/`, `models/`, local source `inputs/`, environments and caches
remain ignored. Preserve tracked dataset references. This branch contains only
production code/configuration, operational docs, synthetic tests, packaging and
CI. No real training, scoring, export, database query or final-test evaluation
was part of this implementation. See [verification](docs/verification.md).

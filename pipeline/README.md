# Development Pipeline

Follow [the lessons and research rules](../docs/research_rules.md) for new
research. Grouped development gains did not transfer to improved October
precision. Preserve the completed results; do not rerun candidates on inspected
October outcomes or treat a research bundle as a promoted operational model.

The project has three phases. The supported runner executes preparation and
model development; final evaluation runs separately under frozen methodology
and explicit authorization. Reusable modeling and
data logic lives in `reseller_churn/`, with orchestration in
`pipeline/development_workflow.py`.

The installed CLI, `python -m reseller_churn --help`, exposes preparation,
grouped development and outcome-free saved-model export. Historical fixed
training, scoring and guarded cutoff evaluation remain separate commands. See
[code architecture](../docs/code_architecture.md) for shared helpers and bundle
compatibility.

| Phase | Entry point | Purpose |
| --- | --- | --- |
| Prepare data | `pipeline/prepare_data.py` | Reconcile explicit clean-order/header/line inputs, enforce declared and loaded coverage, and save ten-feature labeled development rows under one run ID |
| Develop models | `pipeline/develop_models.py` | Select complete configurations and cutoffs using inner reseller groups, assess on outer groups, and save provisional full-development refits |
| Final evaluation | `debug/run_final_testing_evaluation.py`; instructions in `pipeline/12_final_evaluation/` | Authorized frozen-model October evaluation completed; excluded from the supported runner |

## Grouped precision and ablation studies

The latest user-requested studies are standalone research commands in
`comparisons/`; they are not automatic stages of the supported runner.
They maximize precision while requiring pooled recall >90% in each grouped
repetition. Only the approved development CSV and saved development validation
artifacts are inputs. No command below opens or evaluates final testing.

The [repeated grouped study](../comparisons/grouped_precision_recall.md) records
three repetitions of five outer and three inner grouped folds, the eight-cell
search grid, trained research bundle, and verification. The
[component study](../comparisons/grouped_ablation.md) records 14 fixed
configurations, exact reuse, 17 contrasts, adjusted uncertainty, and conclusions.
The latter does not automatically replace the saved candidate.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe -m pytest -c comparisons/grouped_precision_pytest.ini comparisons/test_grouped_ablation.py comparisons/test_grouped_precision_recall.py debug/test_current_model_diagnosis.py -q
.venv/Scripts/python.exe comparisons/grouped_precision_recall.py --development results/development-precision-diagnosis/development_features.csv --preparation-manifest results/development-precision-diagnosis/manifest.json --workers 4 --output results/repeated-grouped-precision-NEW
.venv/Scripts/python.exe comparisons/verify_grouped_precision_recall.py results/repeated-grouped-precision-NEW
.venv/Scripts/python.exe comparisons/grouped_ablation.py --source results/repeated-grouped-precision-NEW --workers 6 --output results/grouped-component-ablation-NEW
.venv/Scripts/python.exe comparisons/verify_grouped_ablation.py results/grouped-component-ablation-NEW
```

Every output directory must be new. The required development feature artifact
and its provenance are produced by the
[development diagnosis preparation](../docs/debugging.md#development-precision-diagnosis--2026-10-16).
The completed local runs are `results/repeated-grouped-precision/` and
`results/grouped-component-ablation/`; each contains a protocol, saved
predictions, metrics, hashes, verification, and a detailed report. Generated
files stay ignored. Do not rerun completed experiments just to read results.

The original study/ablation checks passed 20 focused tests and both standalone
synthetic smoke checks. The subsequent evaluation and diagnosis expanded the
focused suite to **39 passing tests**, last verified on 2026-10-16:

```powershell
.venv/Scripts/python.exe -m pytest -c comparisons/grouped_precision_pytest.ini comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py debug/test_current_model_diagnosis.py debug/test_diagnose_frozen_failure.py -q
```

These are historical verification results, not a test run performed by this
documentation update. The grouped-study verifier also performs five
development-only fidelity refits; it is not a read-only metrics check. Those
copies verify saved-bundle reproducibility and are not additional model-quality
evidence. The
normal repository-wide pytest command encountered a research-package /
merged-final-module collision during those historical runs. The collision is
now resolved by giving frozen configuration, strict preparation, and cutoff
evaluation their own modules within the existing domain packages. Duplicate
TOML tables and initializer conflict markers were repaired earlier.
The [architecture record](../docs/code_architecture.md) describes the repair.
Synthetic smoke checks execution, not model quality.

## Run preparation and development

The supported workflow uses explicit source files and repeated nested grouped
validation. Existing reference datasets and historical artifacts are preserved.
Run from the repository root, using a new run ID:

```bash
python pipeline/run_full_development_pipeline.py \
  --run-id grouped-example \
  --orders inputs/accepted_orders.csv --stores inputs/stores.csv \
  --headers inputs/order_headers.csv --lines inputs/order_lines.csv \
  --coverage-from 2011-01-01 --coverage-through 2013-10-01
```

Coverage is a source-completeness declaration, not a transaction-date estimate.
Raw headers omitted from the accepted clean orders require an independently
reviewed `--approved-excluded-order-ids` list. Unapproved omissions fail.
See the [input contract](../docs/input_contract.md).

Preparation writes `results/<run_id>/preparation/development.csv`. Run grouped
development separately with `python pipeline/develop_models.py --run-id <run_id>`.
The installed CLI supports the same study directly from a development CSV:

```bash
python -m reseller_churn develop \
  --development results/grouped-example/preparation/development.csv \
  --output results/grouped-example/development \
  --models models/provisional/grouped-example
```

For a self-contained execution check, run `python pipeline/grouped_smoke_test.py`.
It prepares synthetic raw files, fits grouped ensembles, reloads a selected model
and exports synthetic scores/flags. It reads no research datasets or outcomes.
The earlier reference-parity and temporal orchestration remain in
`legacy_prepare_data.py` and `legacy_temporal_workflow.py`; they are historical
reproduction entry points and are not invoked by supported commands or CI.

## Search options

Defaults compare original ten-feature histogram boosting with global class
weights and period/class weights, using five model seeds. Three repetitions
(seeds 42, 78, 88) each use five outer grouped folds and three inner grouped folds.
Every snapshot of a reseller stays together at both levels. Fold membership and
row identity are saved. Any infeasible fitting partition fails before fitting;
period/class weighting requires both classes in each of the four fitting periods.

`--outer-folds`, `--inner-folds`, `--repeat-seeds`, `--inner-seed` and `--seeds`
make the validation and model randomness explicit. `--architectures`,
`--feature-sets`, `--imbalance-strategies`, and `--tune` restrict or expand the
candidate specifications. Strict source preparation produces the ten-feature
catalog; other registered catalogs require a separately prepared development CSV
with their columns. Historical searches retain their original catalogs.

Cutoff objectives are explicit:

- `--cutoff-objective recall_at_precision` (default): maximize pooled inner OOF
  recall subject to each requested `--precision-floors` fraction. Ties prefer
  precision, then the higher cutoff. Default floors are 0.1 through 0.9.
- `--cutoff-objective precision_at_recall --minimum-recall 0.9`: maximize pooled
  inner OOF precision subject to recall strictly above 0.9. Ties prefer recall,
  then the higher cutoff. Precision-floor arguments do not apply to this policy.

Candidate ties prefer fewer features, then a stable candidate ID. Outer outcomes
never select the candidate or cutoff. Scores from observed inner OOF values are
used directly; there is no replacement by a two-decimal cutoff grid. These
policies preserve the corresponding research objectives. They differ from the
historical temporal workflow's every-fold precision rule.

Grouped assessment estimates held-out reseller behavior within the development
periods. Historical features still use only information before their snapshot.
A saved model cannot score before all of its training labels became available.

## Read the outputs

| Location | Contents |
| --- | --- |
| `results/<run_id>/development/protocol.json` | Candidates, cutoff policies, split settings and normalized input hash |
| `repeat_*/fold_*/split.json` | Outer row indices and reseller membership, referencing canonical sorted development rows |
| `repeat_*/fold_*/inner_oof.csv` | Training-only OOF scores and inner fold membership for every candidate |
| `validation_predictions.csv` | Outer predictions using inner-selected candidates and cutoffs |
| `selection_status.csv` | Explicit selected/infeasible status for every outer fold and policy |
| `operating_metrics.csv` | Repeat and snapshot recall, precision and false alerts; incomplete outer coverage is flagged |
| `inner_oof.csv` | Full-development inner scores for the provisional final selection |
| `selection/policy_*.json` | Configuration, full-development cutoff, runtime and model checksum |
| `manifest.json` | Input identity, nested evidence checksums and execution provenance |
| `models/provisional/<run_id>/` | One saved ensemble per distinct full-development selection |

A provisional full-development selection is not automatic promotion. Preserve
outer results and infeasible statuses when judging a procedure.

Build prediction features with `features`, supplying the same four source-file
arguments and coverage declarations plus `--snapshot` and `--output`. Then use
an existing grouped selection manifest or supported period/class research bundle:

```bash
python -m reseller_churn export \
  --model results/all-candidate-precision-october-20261018-04/period_class_weighted/frozen_models.joblib \
  --features results/may-snapshot/features.csv \
  --threshold 0.27912545951329487 --output results/may-prediction-export
```

This is a future execution example, not a completed export. The feature directory
must have a matching feature-stage manifest with sufficient declared and loaded
coverage. The proposed 2014-05-02 snapshot requires annual inputs in
`[2013-05-02, 2014-05-02)`; eligibility uses `[2013-11-02, 2014-05-02)`.
Reported transactions through May 1 alone do not certify complete clean orders,
headers, lines or historical availability.

`predictions.csv` contains StoreID, snapshot, n_orders_3m, churn_probability,
threshold, predicted_churn, model_id and rank. Scores are mean probabilities in
the recorded feature order; flags use `>=` before display rounding. This export
currently supports the ten-feature source contract. It fits no model, optimizes
no cutoff and reads no outcomes or adjacent saved prediction files.

## Historical diagnostics and separate research

The numbered scripts are available separately and are not used by the
supported runner. Their preset configurations and result protocols remain
historical. Stages 03 and 04 write generated preparation CSVs to
`results/legacy_preparation/`. Other historical commands retain their own
report locations.

Additional research-only commands:

```bash
python pipeline/06_model_selection/02_compare_model_design_factors.py
python comparisons/validation_schedule_audit.py
python pipeline/09_hyperparameter_tuning/01_tune_models.py
```

These commands use their own research assumptions and do not select the
supported runner's models. See [project context](../docs/project_context.md)
for validation questions and [data analysis](../docs/data_analysis.md) for
source provenance and tracked references.

## Debugging and feature experiments

The earlier feature-comparison research uses root `debug/`, separately from the supported
runner. It compares the seven Pruned features with ten features (Pruned plus
three discount inputs) using temporal validation and stratified folds grouped
by StoreID. The ten-feature set is defined in the debug comparison, not in
the main `FEATURE_SETS` registry; it is not a supported `--feature-sets ten`
option. These experiments write predictions and reports, not promoted model
bundles.

| Script in `debug/` | Purpose | Reports under `results/debugging/` |
| --- | --- | --- |
| `diagnose_april_missed_churners.py` | Describe missed April churners using saved RFM predictions | `april_2013_precision_030/` |
| `compare_rfm_seven_features.py` | Fixed weighted Logistic Regression comparison of RFM and seven inputs | `rfm_vs_seven_features_precision_030/` |
| `inventory_companyx.py` | Read local CompanyX table, column, key, and order-field metadata | `companyx_features/source/` |
| `extract_companyx_history.py` | Retrieve reseller order headers/lines before 2013-04-01 and source profiles | `companyx_features/source/` |
| `build_companyx_features.py` | Preserve existing clean orders, reconcile details, and construct 11 historical candidates | `companyx_features/` |
| `evaluate_companyx_features.py` | Associations and separate basket/discount/diversity comparisons against seven inputs | `companyx_features/` |
| `compare_seven_ten_features.py` | Three architectures, seven/ten inputs, two validation methods, fixed 30% precision floor | `seven_vs_ten_features/` |
| `verify_companyx_outputs.py`, `verify_seven_ten_outputs.py` | Check saved hashes, metrics, populations, and report links without training | Their respective experiment folders |

### Prerequisites and reproduction

The scripts currently use fixed paths rather than a general run-ID interface.
They require the existing preparation/selection artifacts under
`results/codex-luna-verification-20261010-1/` and tracked clean orders. The latest
comparison additionally requires the saved CompanyX feature data, its manifest,
and earlier validation predictions under `results/debugging/companyx_features/`.
Generated inputs are ignored by Git, so a fresh checkout does not contain a
complete runnable experiment. Preserve the original restricted run used by
these scripts; a new preparation run with a different ID is not automatically
selected by them.

SQL extraction requires the existing local `companyx-audit` Docker container,
restored `CompanyX_Audit` database, configured container credential, and Docker
access. `companyx_readonly.py` executes the defined SELECT queries without
printing the credential. Cleaning and model comparisons use saved CSVs and do
not require Docker once extraction is complete. Association screening uses
SciPy in the project environment.

With these prerequisites, rebuild the research artifacts in this order:

```bash
bi_venv/bin/python debug/compare_rfm_seven_features.py
bi_venv/bin/python debug/inventory_companyx.py
bi_venv/bin/python debug/extract_companyx_history.py
bi_venv/bin/python debug/build_companyx_features.py
bi_venv/bin/python debug/evaluate_companyx_features.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python debug/compare_seven_ten_features.py
bi_venv/bin/python debug/verify_companyx_outputs.py
bi_venv/bin/python debug/verify_seven_ten_outputs.py
```

The latest comparison performs 154 fits: deterministic Logistic Regression once
per fold and five-seed ensembles for each tree architecture. Its outputs include
the frozen protocol, fold membership/counts, averaged and individual-seed
predictions, operating selections, threshold grids, subgroup metrics, feature
deltas, paired changes, and hashes. Rerunning a debug script replaces its own
fixed output files, unlike the supported runner's unique-run directory policy.

Debug tests are outside the default `pytest` test path. Both CI workflows run
`python -m pytest -q tests debug` to cover the development and synthetic debug
safeguards without requiring saved experiment outputs or the audit database.
The original development
comparison passed 39 focused tests; the authorized final evaluation adds eight
synthetic safeguards, giving **47 passing focused tests** with the same command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bi_venv/bin/python -m pytest -q debug tests/step_04_evaluation/test_recall_selection.py tests/step_04_evaluation/test_metrics_consistency.py
```

Read [latest metrics](../docs/evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features)
for results and [debugging.md](../docs/debugging.md) for the full evidence chain.
Both validation methods reuse inspected development labels; grouped scores
are not estimates of future-period performance. Protected final evaluation
remains guarded.

## Completed final testing evaluation

The user authorized the 2026-10-11 final test and fixed the ten-feature boosting
model and grouped-development cutoffs before outcome access. The separate
entry point `debug/run_final_testing_evaluation.py` prepared and checkpointed
five refitted models and 492 unlabeled predictions, then verified their hashes
before its label-evaluation stage. It preserves fixed outputs and rejects
overwriting an existing preparation or completed evaluation.

Reports are in `results/final_evaluation/gb10_grouped_20261011/`, with the model
bundle in `models/provisional/gb10_grouped_20261011/ensemble.joblib`. Primary
recall is 93.48%, but precision is 28.48%, below the intended 30%. It catches
43 of 46 churners, misses three, and produces 108 false alarms. The October
test outcomes are now inspected and remain excluded from development training
and selection. See [Stage 12](12_final_evaluation/README.md) and
[final results](../docs/evaluation_metrics.md#final-test-evaluation--2026-10-11).

Use the saved-output verifier rather than rerunning the completed final test:

```bash
bi_venv/bin/python debug/verify_final_testing_outputs.py
```

## Persistent-error research continuation

The development-only matched-history investigation and single-feature nested
comparison are separate from the operational workflow. Read the
[exploration protocol and findings](../comparisons/persistent_error_investigation.md),
[frozen model-comparison protocol](../comparisons/basket_breadth_protocol.md), and
[current journal](../docs/research_journal.md) first. Each output directory must
be new. The comparison fits 600 estimators; its verifier fits none.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.persistent_error_investigation --output results/persistent-errors-NEW
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.basket_breadth --histories results/persistent-errors-NEW --workers 1 --output results/basket-breadth-NEW
.venv/bin/python -m comparisons.verify_basket_breadth results/basket-breadth-NEW
```

Defaults use the relocated source study, the reproduced Linux diagnosis, and
the exact historical-byte order copy described in the journal. Override
`--orders` and `--diagnosis` for a different verified location. The strict audit
rejects a different order hash; LF/CRLF checkout differences must be investigated,
not bypassed by rewriting a historical manifest. Neither command opens October
outcomes or produces a replacement operational model.

The bounded annual-breadth contingency and the separately frozen cadence
stability check use the same fitting utilities:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.basket_breadth --feature unique_products_12m --histories results/persistent-errors-NEW --output results/annual-breadth-NEW
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.cadence_stability --output results/cadence-stability-NEW
.venv/bin/python -m comparisons.verify_basket_breadth results/cadence-stability-NEW
.venv/bin/python -m comparisons.breadth_reporting results/cadence-stability-NEW
```

Read [the cadence stability protocol](../comparisons/cadence_stability_protocol.md)
before execution. New grouped seeds are a stability check on reused development
rows, not independent confirmation. The reporter creates a new report and refuses
to replace an existing one. Live CompanyX availability checks and the restored
container/query access are recorded in
[data analysis](../docs/data_analysis.md#live-source-availability-follow-up--2026-10-16).
Do not rerun old fixed-output extraction scripts just to inspect the source;
new audits must use new directories and preserve historical exports.


## Development research preparation — 2026-10-17

Use [the development preparation protocol](../comparisons/development_research_protocol.md)
for future matched experiments. It separates the existing grouped and temporal
views,documents current data guards,and requires frozen period/activity harm
limits alongside the original recall/false-alert objective. It does not introduce
a candidate,change historical results or automatically execute training/testing.

The supported no-fit review of saved fixed-numeric studies is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m comparisons.review_development_study \
  --source results/cadence-unit-revenue-20261017 \
  --output results/CHOOSE-A-NEW-development-review-directory
```

This reads only declared canonical development artifacts and saves complete
paired period/repeat/fold/activity reports,sorted harm flags,split timing and
saved-cutoff inner/outer operating tables. It never loads a model,fits,selects a
cutoff or calls a final evaluator. Existing output directories are rejected.
The final verified preparation run is results/development-preparation-review-20261017-verified;
its original manifest retains the code identities at execution. Shared assessment
now emits paired_operating_metrics.csv and worst_slices.csv for future executions.
An optional EXPLICIT period_limits argument activates prospective acceptance
checks; omitting it preserves historical behavior. All new experiments must freeze
and pass their limits as specified by the preparation protocol.

Verification functions vary: synthetic smoke tests may fit synthetic models;
historical model verifiers may refit real development models. Inspect an entrypoint
before running it. The review command above is the no-fit route for these reports.

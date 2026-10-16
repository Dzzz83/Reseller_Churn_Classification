# Continue work from another machine

Use branch `pipeline_preparation`. This checkpoint includes the import repair,
baseline diagnosis implementation/tests, current research rules, aggregate
report, and the local research/data/presentation archives approved for publication.

## Set up

```bash
git clone --branch pipeline_preparation https://github.com/Dzzz83/Reseller_Churn_Classification.git
cd Reseller_Churn_Classification
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r handoff/2026-10-16/requirements-runtime.txt
python -m pip install -e ".[dev]"
python handoff/2026-10-16/restore_artifacts.py
```

The recorded runtime is Python 3.12.14. The requirements snapshot records the
principal package versions used here. Recreate the environment rather than
copying `.venv`. Old serialized models may require their recorded exact runtime
and source code; restored bundles are preserved evidence, not automatically
compatible models for the repaired fixed-model loader.

The restore script checks all archive/member SHA-256 values and validates paths.
It skips identical existing files and refuses to overwrite different local files.
Use `--verify-only` to check downloaded archives without extracting them.

## What is included

| Archive | Contents |
| --- | --- |
| `artifacts/research-artifacts.zip` | Saved development studies, predictions, manifests, candidate bundle, diagnoses, smoke-study artifacts, source-regeneration results, and the packaging feature-parity record |
| `artifacts/source-inputs.zip` | Preserved order header/line exports, raw export CSVs, SQL queries, and source manifest |
| `artifacts/presentation-artifacts.zip` | Existing presentation outputs, working assets/scripts, and paper-review images |

Each archive restores its original `results/` or `inputs/` paths. Nothing extracts
into tracked code. Credentials, local agent settings, virtual environments,
caches, temporary tests, and disposable packaging-build copies are excluded.
The [archive manifest](artifacts/manifest.json) enumerates every preserved file.
Historical manifests and model files are unchanged. Some historical scripts
contain machine-specific absolute paths; do not rewrite signed/checksummed
historical records to make them match another computer. The baseline diagnostic
resolves its original inputs under the new sibling `results/` directories while
still verifying their original hashes.

## Current state and next task

The fixed CLI and research packages coexist without naming collisions. The
final combined suite passed **210 tests**, including relocation and archive
restore safeguards. All 566 archived files passed checksum verification; a
fresh extraction reproduced the 1,375-case, 15-fold baseline provenance audit
from the moved paths. No real models were trained or October outcomes accessed
by the diagnosis or handoff checks.

Read [the aggregate diagnostic report](baseline-report.md) and
[diagnostic methodology](../../comparisons/baseline_error_diagnosis.md).
The full report/data is restored under
`results/baseline-error-diagnosis-20261016-verified/`.

The current objective is **fewer false alerts while matching or exceeding baseline
recall on the same development validation splits**. The immediate next task is
to compare persistent development errors with similar correctly classified cases
to identify missing historical signals. No new feature or candidate is selected.
October outcomes are inspected and must not select the next features or cutoff.

## Checks and diagnostic reproduction

```bash
python -m reseller_churn --help
python -m pytest -q tests debug comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py
python -m comparisons.baseline_error_diagnosis --output results/baseline-error-diagnosis-NEW-MACHINE
```

Always choose a new output directory. Reproducing this diagnostic reads saved
development predictions; it performs no model fitting. Generated report manifests
record the current local paths, code, and runtime, so new manifest bytes will
differ from the historical originals even when diagnostic tables agree.

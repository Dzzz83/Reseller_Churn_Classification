# Code architecture and import repair

The repository supports fixed ten-feature scoring and model-development research.
Both use the same domain packages. Explicit module names distinguish frozen
behavior from candidate selection; package initializers contain no implementation
or broad re-exports.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Parse fixed-workflow commands and call their implementations |
| `config/frozen_model.py` | Frozen feature order, parameters, seeds, training snapshots, and original cutoff roles |
| `data/preparation.py` | Validate coverage, reconcile accepted orders with headers/lines, and build the ten historical features and approved development rows |
| `model.py` | Train/load the fixed ensemble and save ranked scores |
| `evaluation/cutoff_evaluation.py` | Validate separately supplied scores/outcomes and report explicit cutoffs without optimization |
| `data/prediction_window.py` | Define history, eligibility, and outcome boundaries |
| `data/eligibility.py`, `data/churn_labels.py` | Identify eligible resellers and construct historical labels |
| `features/feature_pipeline.py` | Build the research workflow's 17-feature catalog |
| `modeling/` | Define, compare, and refit research candidates |
| `evaluation/metrics.py` | Shared ranking and threshold calculations |
| `evaluation/selection.py` | Research selection under precision constraints |
| `artifacts.py` | Hash files, write JSON/manifests, and protect immutable output directories |
| `pipeline/development_workflow.py` | Orchestrate temporal candidate comparison and provisional refits |
| `comparisons/`, `debug/` | Separate historical studies and diagnostics |

Example imports identify their owner directly:

```python
from reseller_churn.config.frozen_model import TRAINING_SNAPSHOTS
from reseller_churn.data.preparation import build_features, load_sources
from reseller_churn.evaluation.cutoff_evaluation import evaluate
```

Previously, `config.py`, `data.py`, and `evaluation.py` shared names with package
directories. Python selected the packages, hiding the functions/constants in the
files. Those files were moved into the packages and their callers updated.
The CLI command names and arguments are unchanged.

## Shared behavior and objects

Preparation reuses `PredictionWindow`, reseller eligibility, and historical churn
labels. It still checks declared source coverage and reconciles accepted orders
before feature construction. These safeguards are separate from the historical
research CSV loader's contract.

Cutoff reporting calls the shared metric functions while preserving its report
columns and subgroup definitions. Research and fixed-workflow callers use the
same file-hashing and JSON-writing helpers. Separate research configurations
remain explicit: changing a research default must not change the frozen method.

`PredictionWindow` encapsulates date boundaries, `Sources` holds reconciled
inputs and coverage, and `FixedEnsemble` holds fitted estimators. Transformations
and orchestration remain functions. There is no additional service hierarchy.

## Saved-model compatibility

`FixedEnsemble` remains at `reseller_churn.model.FixedEnsemble`, preserving its
serialized class location. Moving the three conflicting modules still changes
the implementation identity, so old bundles are not automatically compatible.

New manifests declare `source.package_hash_format = relative-python-paths-v1`.
Their `source.package_sha256` maps every package Python file's relative path to
its SHA-256, including nested modules. Paths distinguish files such as the
different `__init__.py` files. Loading requires the same hash format, source
hashes, runtime versions, configuration, and output checksums.

Legacy manifests without this format are rejected with an explicit message.
Use their recorded package/environment to reproduce them, or explicitly refit
under the repaired package. Preserve old bundles and manifests unchanged; no
manifest rewriting or relaxed integrity checks are part of this repair.

## Verification

Install the declared dependencies with `python -m pip install -e ".[dev]"`.
Run the combined regression checks and standalone research safeguards:

```bash
python -m reseller_churn --help
reseller-churn --help
python -m pytest -q tests debug comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py
python pipeline/smoke_test.py
```

The tests cover time boundaries, source reconciliation, reference features and
labels, weighting, seed averaging, ranking, CSV precision, cutoff equality,
artifact tampering, nested source changes, and legacy identity rejection.
The synthetic smoke exercises all five CLI stages. An additional installed-wheel
check from outside the repository verifies packaging rather than checkout imports.
These checks establish execution and behavioral consistency, not model quality.
They do not rerun historical October evaluations or select new model settings.

### Completed local checks — 2026-10-16

- The combined command above passed **191 tests** on Python 3.12.14.
- Both CLI help entry points passed. A clean wheel installed into an isolated
  target passed all five synthetic stages from a working directory outside the
  repository; imports were asserted to resolve to that installed package.
- Rebuilt development inputs matched the saved development reference for all
  1,375 account/snapshot identities and 212 churn labels exactly. All ten feature
  columns agreed within absolute tolerance `1e-10`; maximum difference was
  `5.82e-11`. No real-data models were fitted or final outcomes accessed.
- Changed Python files passed Ruff checks, with the existing standalone-script
  path-setup convention excluded from E402 checks. Modified core modules and
  fixed-workflow tests passed Ruff formatting checks.

Generated verification files are ignored under
`results/packaging-fix-verification/`. Existing stale `build/` output was preserved
there as `preexisting-build/` before rebuilding, so deleted modules cannot leak
into the new wheel through that local cache. Historical reports, model bundles,
and tracked dataset references were not overwritten. GitHub Actions were not run
as part of these local checks.

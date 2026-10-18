# Code architecture and import repair

The repository supports fixed ten-feature scoring and model-development research.
Both use the same domain packages. Explicit module names distinguish frozen
behavior from candidate selection; package initializers contain no implementation
or broad re-exports.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Parse preparation, grouped-development, saved-scoring and historical fixed commands |
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
| `pipeline/development_workflow.py` | Thin entry point for supported grouped development |
| `modeling/grouped_validation.py` | Repeated reseller splits and explicit cutoff objectives |
| `modeling/grouped_development.py` | Inner selection, outer assessment and provisional refits |
| `modeling/grouped_command.py` | Shared command arguments and development CSV boundary |
| `modeling/weighting.py`, `modeling/ensemble.py` | Training-only period/class weights, probability averaging and threshold decisions |
| `modeling/bundles.py`, `prediction.py` | Validate saved research/grouped bundles and export scores/flags without outcomes |
| `data/commands.py`, `data/validation.py`, `features/validation.py` | Shared preparation orchestration and source/feature contracts |
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

`PredictionWindow` encapsulates date boundaries. `Sources` holds reconciled
inputs, declared coverage and the effective loaded cutoff. The existing
`Candidate` describes a complete model specification; `GroupedValidationPlan`
holds repeat/fold settings and `ThresholdPolicy` names the selection objective.
`FixedEnsemble`, `FittedProbabilityEnsemble` and the read-only `LoadedEnsemble`
retain fitted estimators at explicit workflow boundaries. Shared calculations
and orchestration remain functions; no inheritance hierarchy is required.

Historical temporal and reference-parity orchestration is retained in
`pipeline/legacy_temporal_workflow.py` and `pipeline/legacy_prepare_data.py`.
Supported commands and default CI do not invoke it. Historical comparison entry
points remain independent, while grouped splits, cutoff scans, strict recall
selection and period weights use shared package implementations.

## Saved-model compatibility

`FixedEnsemble` remains at `reseller_churn.model.FixedEnsemble`, preserving its
serialized class location. Moving the three conflicting modules still changes
the implementation identity, so old bundles are not automatically compatible.

Historical fixed-model manifests declare `source.package_hash_format = relative-python-paths-v1`.
Their `source.package_sha256` maps every package Python file's relative path to
its SHA-256, including nested modules. Paths distinguish files such as the
different `__init__.py` files. Loading requires the same hash format, source
hashes, runtime versions, configuration, and output checksums.

Legacy manifests without this format are rejected with an explicit message.
Use their recorded package/environment to reproduce them, or explicitly refit
under the repaired package. Preserve old bundles and manifests unchanged; no
manifest rewriting or relaxed integrity checks are part of this repair.

The retained period/class research adapter accepts only the `{spec, models}`
contract for the original ten-feature HGB configuration. It checks the model-file
checksum, runtime versions, frozen configuration identity, feature order, binary
class order, estimator parameters and five seed identities. It reads only the
model and its manifest, not adjacent score files. Unlike the historical fixed
loader, this explicit adapter supports the refactored implementation without
requiring equality of every package source file. Original source metadata is
carried into new exports; original bundles/manifests are never rewritten.

New grouped selections use `grouped-selection-v1`. Their manifests record the
candidate, cutoff, validation plan, runtime, data hash and relative model path.
Loading checks the selection checksum against run evidence and checks the model
checksum, candidate, data identity, feature/class order and estimator settings.
Keep result and model directories in the same relative layout when moving them.
An export cannot precede the saved training-label availability date.

## Verification

Install the declared dependencies with `python -m pip install -e ".[dev]"`.
Run the self-contained default suite and grouped smoke:

```bash
python -m reseller_churn --help
python -m pytest -q
python pipeline/grouped_smoke_test.py
```

Default tests use synthetic inputs. They cover grouped separation, inner-only
selection, infeasible partitions, source boundaries, loaded coverage, explicit
exclusions, source/feature consistency, serialization, seed averaging, exact
cutoff equality, CSV precision and tampering. Historical/reference-data tests
remain accessible by explicit paths. CI also runs focused synthetic compatibility
tests for the historical grouped research entry points. None of these checks
establish model quality or require protected evaluation artifacts.

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

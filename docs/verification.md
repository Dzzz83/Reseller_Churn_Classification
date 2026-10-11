# Final pipeline verification and review

Verified on 2026-10-11. Execution branch: `final-pipeline-preparation`.
Base: refreshed `origin/main`, commit
`b43e44d787236d77dc46fd9acc26b9ec831b3544`. Research source:
`pipeline_preparation`, commit `f53202dbe686a58b9f37b43d7da0b9f6667180c6`.
The research checkout, original scripts, and evaluation artifacts were preserved.

## Completed checks

| Check | Result |
| --- | --- |
| Final suite on Python 3.11.16, fresh dependencies | 44 passed |
| Final suite on original Python 3.14.7 environment | 44 passed |
| Synthetic standalone end-to-end smoke | Prepare, train, features, predict, and cutoff evaluation passed |
| Installed package outside the repository | All five CLI stages passed with synthetic CSVs and outcomes; imports resolved to site-packages |
| Development feature/label parity | All ten features and labels match 1,375 saved development rows, within 1e-10; parsed training feature values also match exactly |
| Original final feature parity | All ten features match 492 saved rows, within 1e-10 |
| Final prediction parity | All 492 serialized probabilities match exactly after round-trip parsing |
| Original cutoff decisions | All decisions match at 0.01, 0.06, 0.55, 0.88 |
| Existing final metric parity | All 12 overall/subgroup metric rows match within 1e-12 |
| Original evidence integrity | All 32 recorded original research/evaluation file hashes unchanged; original final verifier passes |
| Static checks | Ruff lint/format, Python 3.11 syntax, and workflow YAML passed |
| Repository checks | No research imports, generated outputs excluded, tracked datasets unchanged, diff and local Markdown links checked |

Parity training/scoring used the original Python 3.14 environment. Clean install
and smoke checks used Python 3.11.16 with NumPy 2.4.6, pandas 3.0.6,
scikit-learn 1.9.1, and joblib 1.6.0. Model loading requires matching saved
runtime versions and package hashes; cross-version model portability is not
assumed. Original artifacts are retained locally outside Git, with local parity
records under `results/parity/` in this checkout.

An initial parity check found differing scores for 15 accounts despite matching
aggregate metrics. Both ensembles gave identical predictions on the same
in-memory features. The difference was default CSV float parsing moving some
feature values across tree bin boundaries. Round-trip parsing restored the
original scores. Regression tests now cover scoring at a model boundary and
evaluation at a score exactly equal to a cutoff.

The saved October outcomes were reused solely to verify the extracted behavior
and reports. No protected outcome extraction, cutoff optimization, model
selection, or new held-out evaluation occurred. Synthetic smoke results are
execution evidence, not model-quality estimates. GitHub-hosted Actions have
not run during this local preparation.

## Main-branch scope

Included: final configuration, source reconciliation, historical features,
approved development preparation, fixed ensemble training/loading, ranked
prediction, explicit cutoff reports, focused tests, package installation, CI,
and final-purpose documentation.

Excluded: architecture/feature/imbalance searches, tuning grids, alternative
validation experiments, debugging investigations, dated experiment directory
dependencies, historical numbered runners, research-only tests and CI, raw
local exports, and generated model/report objects.

The order-header and order-line exports are explicit required inputs. Main's
tracked source CSVs do not contain all discount-feature facts. Source coverage
and independent evaluation chronology require user/source declarations; they
cannot be proven solely from the provided CSVs. Original order-time availability
and purchase/cleaning assumptions remain in effect.

## Suggested pull request

Title: Add final ten-feature churn pipeline with configurable cutoff evaluation

Body:

> Add a self-contained fixed-model pipeline that prepares historical features,
> trains the five-seed gradient-boosting ensemble, saves ranked probabilities,
> and reports user-supplied score cutoffs without retraining or threshold search.
> Research searches and diagnostics remain on pipeline_preparation.
>
> Verified 44 tests on Python 3.11 and 3.14, a fresh installed-package smoke,
> feature parity, exact parity for 492 original predictions, and all 12 original
> metric rows. Original research/evaluation artifacts remain unchanged.
>
> October outcomes are already inspected; further cutoff reports are
> descriptive. Original primary precision is 28.48%, below the intended 30%.
> Header/line CSV exports are required for discount features.

Review the diff against main and the documented CLI/input contracts. The user
will push the branch, create the PR, and decide whether to merge. No PR creation,
push, or merge is part of this preparation.

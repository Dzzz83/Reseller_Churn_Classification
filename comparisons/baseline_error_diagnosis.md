# Baseline error diagnosis

This development-only diagnostic describes the original ten-feature model's
saved held-out predictions. It fits no models and selects no new cutoffs.
Its purpose is to locate consistent false alerts, missed churners, ranking
overlap, and decision variation before proposing new features.

## Inputs and verification

The default source is `results/repeated-grouped-precision/`: three repetitions
(42, 78, 88), five outer reseller-grouped folds, 1,375 account/snapshot cases,
422 resellers, and 212 development churn labels. The diagnostic consumes only
the original baseline's two policies. Other candidate predictions are filtered
out and not compared. It does not open October evaluation files or outcomes.

Input checks verify study/preparation manifests, consumed file hashes, the
ten-feature baseline specification and seeds, development-only coverage,
reseller-isolated outer/inner memberships, saved inner-cutoff metrics, and
outer training-content hashes. All 108 saved baseline fold/repetition metric
rows must reproduce. No verifier that performs refitting is invoked.

The original study read training features with pandas' default float parser.
This diagnostic reconstructs that parser when auditing training content, and
uses round-trip parsing for serialized study inputs and predictions. This
distinction preserves fitted values rather than silently changing their
binary representation. Source completeness and fitted estimator internals are
not freshly certified by this metadata audit.

When moving the checkout, recorded absolute input paths are resolved against
the sibling directories under the new `results/` location. Historical manifests
remain unchanged and the original SHA-256 checks still apply.

## Fixed analysis definitions

- Policies: historical 0.06 and saved inner-selected high-recall cutoffs. The
  latter maximize training OOF precision with recall >90%; they are not the
  original 30%-precision policy. Both policies have identical scores/ranking.
- Slices: recent purchase (three months), annual orders (1, 2, 3+), observed
  twelve-month history span (0–90, 91–180, 181+ days), each discount feature's
  missingness, and each development snapshot. Span is last minus first purchase
  within the window, not actual relationship tenure.
- Ranks: descending midrank within the outer validation fold, transformed to
  `(midrank - 0.5) / fold_size`; smaller means nearer the top. Score ties stay
  together. Rank bands are top 10%, 10–25%, 25–50%, 50–75%, and bottom 25%.
- Review flags: within five percentile points of the alert boundary, false
  alerts in the top 25%, and missed churners in the bottom 25%.
- Descriptive recall points: 90%, 95%, and 100%, including ties. These use
  held-out outcomes to characterize curves and must never become fitted cutoffs.
- Small groups: rows <30, churners <10, or non-churners <10. Undefined rates
  remain missing; confusion counts remain explicit, including empty slices.
- Recurrence: errors on 3/3 repetitions are consistent, 2/3 recurring, 1/3
  occasional. Reseller/snapshot identity is preserved. Repeated predictions
  and a reseller's different snapshots are dependent.
- Rank variation of at least 20 percentile points and changing decisions are
  descriptive flags. Comparison pools also change across folds; this does not
  isolate model instability. No confidence intervals or causal claims are made.

Reports retain per-fold/per-repeat metrics. Pooled ranking curves explicitly
combine different fitted models; average fold ranking metrics are also shown.

## Reproduction

Run from the repository root with the declared dependencies installed:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q tests/step_04_evaluation/test_baseline_error_diagnosis.py
python -m comparisons.baseline_error_diagnosis --output results/baseline-error-diagnosis-NEW
```

Every output directory must be new. It contains `report.md`, three plots,
account-level features/profiles/decisions, subgroup and rank-band metrics,
ranking curves, diagnostic high-recall burden, recurrence tables, protocol,
verification, consumed-input/implementation identities, and output checksums.
Generated account-level data remains ignored by Git. A fresh checkout requires
the locally preserved source-study artifacts; tests use synthetic cases.

Implementation is separated into input auditing, analysis transformations,
and reporting under `comparisons/baseline_diagnosis/`; the command module only
orchestrates them. Tests cover ties, cutoff equality, historical boundaries,
profile/missingness boundaries, undefined groups, varying pooled decisions,
dependent repetitions, split leakage, changed provenance, and output preservation.

The next research objective is fewer false alerts while matching or exceeding
baseline recall on the same splits. This diagnostic selects no features or
models and does not establish any replacement's performance.

## Completed diagnosis — 2026-10-16

The completed local report is
`results/baseline-error-diagnosis-20261016-verified/report.md`. These results
describe the saved grouped development predictions, not October testing.

| Operating policy | Mean repeat recall | Mean repeat precision | Mean false alerts per repetition |
| --- | ---: | ---: | ---: |
| Historical fixed 0.06 | 77.36% | 32.96% | 333.67 |
| Saved inner-selected high-recall | 91.51% | 27.16% | 520.67 |

The same scores underlie both rows. The adaptive operating policy catches
30 additional churn cases per repetition on average at the cost of 187
additional false alerts; this is a cutoff effect, not a ranking improvement.

Under the saved high-recall policy, recently purchasing accounts average
85.12% recall, 18.38% precision, and 41.61% false-positive rate. All 91 churn
cases without a recent purchase are caught in every repetition, but so are all
63 non-churn cases without a recent purchase. Those accounts remain eligible
under the six-month rule; "no recent purchase" does not mean already labelled churn.

Errors are not all boundary accidents: 416 account/snapshot cases are falsely
alerted in all three repetitions, including 353 recently purchasing cases;
11 churn cases are missed in all three, all recently purchasing. Sparse
histories alone do not explain misses: 46 of 54 missed-churn events across the
three repetitions have at least three annual orders. Those 54 events are
dependent repeated predictions, not 54 independent churners.

383 of 1,375 cases have rank ranges of at least 20 percentile points across
repetitions. Changes in both fitted models and validation comparison pools
contribute; this does not isolate model instability. April's descriptive
high-recall slice has a mean false-positive rate of 59.66%, versus 37.62–42.53%
in the other snapshots. No temporal-generalization claim follows from this.

The next investigation should compare the histories of persistent errors
with similar correctly classified accounts, including the smaller no-recent-
purchase false-alert group. The diagnosis identifies cases and questions;
it does not prove a cause or select a new feature.

Verification passed 13 new synthetic tests and **204 combined repository,
debug, and standalone-study tests**. All 108 saved baseline metric rows and
15 outer splits reproduced. A separate read-only audit checked all 8,250
case/policy/repetition decisions, all 30 rank groups, all 36 ranking curves
against scikit-learn, all 2,750 case/policy recurrence rows, and all 12 output
checksums. Its local record is
`results/baseline-error-diagnosis-20261016-independent-check.json`.
All three plots were visually inspected. Ruff checks and formatting passed
for the new implementation and tests. No models were fitted, cutoffs selected,
or October outcomes accessed.

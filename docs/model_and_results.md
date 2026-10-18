# Production model and development validation

The supported model is `period_class_weighted`, using the ten historical features
in [the input contract](input_contract.md). Research alternatives, personalized
recency, candidate searches and historical performance reports stay on research
branches. This implementation makes no new performance claim.

## Frozen artifact and cutoff

- Five HistGradientBoostingClassifier estimators, seeds 42, 78, 88, 1034, 2026.
- learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20,
  l2_regularization=0, early_stopping=False; other constructor defaults are
  checked against the pinned estimator runtime, including class_weight=None.
- Native missing-value handling and arithmetic mean of positive-class
  probabilities; ordered estimator classes must be [0, 1].
- Equal total weight in each development period × outcome-class cell, computed
  only from the current fitting partition: weight = n / (8 × cell_count).
  There is no additional class-weight multiplier.
- Approved training snapshots: 2012-07-01, 2012-10-01, 2013-01-01, 2013-04-01.
  All six-month training outcomes resolve by 2013-10-01; earlier prediction
  snapshots are rejected.
- Retained cutoff: **0.27912545951329487**, inclusive `score >= cutoff`.
  The shortened display value 0.27912546 must not be used for decisions.

The retained `frozen_models.joblib` has SHA256:

```text
6ee382620eeae80bb2fce68734813e0323b7620a56c4b73f867ecfdad66894ba
```

Public features and exports use `discount_frequency_12m`. The retained bundle
still records `discounted_line_share_12m`; the loader maps only that name at the
estimator boundary, preserving feature order, values and the binary checksum.
Future development artifacts use the public name directly. Export manifests
record both public and estimator feature orders and any name mapping.

Its dictionary contains `spec` and `models`; it requires no research classes to
load. The production loader validates checksum before deserialization, exact
runtime, specification, feature order, estimator type, all constructor parameters,
seeds, fitted state and binary classes. A manifest cannot change the retained
artifact's cutoff. Runtime: Python 3.14.8, NumPy 2.3.5, pandas 3.0.1,
scikit-learn 1.9.1, joblib 1.6.0. Compatibility is explicit, not inferred from
successful unpickling. No package-history hashes or adjacent scores/results are
required. Keep binaries and source exports outside Git.

The minimal `production-model-v1` manifest records artifact checksum, runtime,
features, parameters, seeds, weighting, aggregation, training snapshots,
label-availability date and exact threshold. The retained manifest ships as
package data. Future development writes the same manifest schema separately
from its local bundle; explicit `--model-manifest` selects it. Only load trusted
pickle/joblib data; hashes verify identity rather than authenticity.

## Future grouped development

`develop` uses one model configuration. The defaults are five outer folds,
three outer repeat seeds (42, 78, 88), three inner folds and inner seed 78.
StratifiedGroupKFold groups by StoreID, so every snapshot of a reseller stays in
one partition. Each fitting partition must contain both classes in all four
periods; infeasible folds fail before fitting.

Inner pooled OOF scores choose the cutoff maximizing recall subject to precision
>=40%. Feasibility uses exact integer TP/alert ratios. Ties prefer precision,
then the larger cutoff. Candidates are the observed scores plus zero, using
inclusive comparison. A missing feasible cutoff remains explicit; there is no
fallback rule. Each outer fold assesses a cutoff selected using only its training
resellers. Full-development inner OOF then selects a future artifact's cutoff,
and five models fit all development rows. The retained artifact/cutoff remains
unchanged.

Development outputs record OOF scores, outer selections, grouped decisions and
repeat/snapshot recall, precision, confusion counts and false alerts. A repeat
with any infeasible outer selection is marked incomplete; metrics over remaining
rows must not be treated as complete assessment. Zero-denominator precision and
recall are reported as zero. Split plans, source provenance, data identity,
runtime and output hashes support reproduction. Grouped validation estimates
unseen-reseller generalization within development periods, not future-period
performance. There is no temporal validation or final-test evaluation command.

## Evidence and readiness limits

The supplied handoff records development-side reconciliation of historical
features, eligibility, development labels, grouped splits, period/class weights,
model parameters and the exact retained cutoff. It found no evidence requiring
retraining. That evidence is historical context; it was not rerun on real data
in this implementation. No protected final-test artifacts were used for model
selection or verification here.

For proposed snapshot 2014-05-02, complete source coverage must contain
[2013-05-02, 2014-05-02), including the six-month eligibility period beginning
2013-11-02. The separate authorized backup check confirmed 467 eligible resellers
and reconciled the complete source window; see [source readiness](verification.md#remaining-export-readiness).
The older detailed exports stopping at 2013-10-01 must not be used for this
snapshot. The authorized 2014-05-02 prediction export has been completed and verified; no
future outcomes were used. Scores are not validated calibrated
individual risk estimates. New independent evaluation requires frozen methodology
and explicit user approval; protected final-test data stays outside development.

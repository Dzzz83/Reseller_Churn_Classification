# Saved-score transfer diagnosis — frozen cycle 4 definitions

2026-10-16; planned while the annual-breadth trial runs, before reading its
outcomes. Run only after both bounded trials complete. This is a no-fit,
no-selection development diagnostic, not a new candidate or cutoff comparison.

Question: how much of the observed decision variation can be reproduced by
changing saved training-selected cutoffs on fixed scores, and how different are
inner OOF operating results from held-out outer results? These calculations
cannot certify transfer to a full-data fit or isolate a causal failure mechanism.

Inputs are only the two newly completed breadth-study directories, preserving
all three repeats, five folds and all 1,375 cases. Verify manifests first. Baseline
is reported once because both runs use identical baseline data/procedure/runtime.
Require baseline predictions/cutoffs to agree between runs. No October data.

For each arm and outer fold, report saved inner OOF precision/recall/alert rate,
outer precision/recall/alert rate/FPR, selected cutoff, training/validation counts,
and seed range. Pool outer confusion counts per repeat; average inner metrics
only as labelled descriptive fold averages. Inner selection optimizes those
metrics, so the inner–outer difference includes selection optimism, population
and fitted-score changes; it is not pure calibration error.

For each unordered pair of repeats and case, hold the score from repeat A fixed
and compare decisions at the two saved case-specific cutoffs. Then hold A's
cutoff fixed and compare decisions at the two saved scores. Repeat with B fixed
and average the two directions. Report actual cross-repeat disagreement plus
these score-only and cutoff-only disagreement rates, overall and by activity.
They are not additive causal shares: interactions and cancellations occur, and
cutoffs came from different fitted score scales. Do not use cross-swapped
cutoffs to compute a winning operational rule. No labels select new cutoffs.

Conditional uncertainty from the model studies does not cover these sources of
variation, repeated adaptive research, source incompleteness or a future cohort.
No new predictor is justified solely by a score-scale or decision-variation flag.
If simple transaction-breadth hypotheses fail, the next data request should be
specific: recording-time/order-validity evidence and prospectively timestamped
relationship/service/payment events or an untouched cohort, not mutable current
profiles or future-derived purchasing aggregates. This diagnosis does not prove
that all transaction-only methods are exhausted or that new data will succeed.

## Product-mix follow-up — frozen before transfer summaries (2026-10-17)

The SKU trial fails complete acceptance despite lower descriptive within-fold FP
burden at90/95/100% recall. Do not select an outer cutoff from those curves. Use
the same no-fit transfer/decision-variation analysis for original baseline,
cadence-no-count and cadence+product mix on matched301/919/1201 splits. Preserve
all15 inner/outer operating rows per procedure and all pairwise repeat variation,
including negative/reversing evidence. Add class-specific score quantiles for
inner OOF versus outer validation and per-inner-fold operating metrics at the
already saved cutoff. These are descriptions of different fitted models/pools,
not calibration evidence, causal shares, independent uncertainty or deployable
cutoffs. No new fit/threshold selection and no October data. A calibration or
model remedy needs separate motivation/protocol, not automatic reaction to a
favorable curve. Original 0.06 and October recall are never tuning targets.

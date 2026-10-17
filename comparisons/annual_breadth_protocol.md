# Annual product breadth — contingent cycle 3 protocol

Frozen 2026-10-16 while the last-order-breadth study is still running, before
reading its outer predictions or interpreting any candidate results. Execute
only if that candidate fails its frozen acceptance requirements. Preserve that
failure and all previous protocols. This is a second bounded hypothesis from
the same matched exploration, not an unlimited search.

## Rationale and procedure

Annual unique-product count is higher for persistent recent false alerts than
similar caught churners in all four snapshot medians, and lower for persistent
misses than correctly unalerted controls in all represented snapshot medians.
The respective overall matched median differences are +1 and -7.33 products;
these patterns survive one/three/five-neighbor medians. Existing basket/diversity
bundles failed a linear model's old temporal objective; annual unique products
have not been isolated in the repeated grouped ten-feature boosting baseline.
This remains a weak exploratory motivation, not a validated association.

Use only baseline ten features plus `unique_products_12m`: count distinct
ProductID values over accepted order lines attached to accepted orders in
[t-12 months,t). No last-order feature is included, so this isolates broader
history rather than combining features after inspecting a favorable fit.

All data, model parameters, seeds, natural populations, grouped outer/inner
partitions, training-only cutoff selection (>90% OOF recall, maximize precision;
ties recall then cutoff), output preservation and reporting are exactly those
in `basket_breadth_protocol.md`. Both arms are fitted in the same current
runtime. No hyperparameter or other feature alternatives are searched.

## Acceptance and uncertainty

Require candidate recall >= baseline in every repeat, FP strictly lower in
every repeat, at least 5% mean FP reduction, and no more than 5 percentage points
loss in mean recall within either activity group or any snapshot. The practical
and subgroup margins remain explicit research defaults.

Use 5,000 paired reseller-cluster draws, seed 2026, carrying all snapshots and
repetitions together. For this second related trial, use the **1.25th percentile**
for the lower recall-difference bound (must be >=0) and FP-reduction bound (must
be >0): Bonferroni across two endpoints and two bounded trials, joint one-sided
95% for this pair of trial decisions conditional on fits. Report 1.25–98.75%
intervals and the earlier trial's original 2.5–97.5% intervals unchanged. This
does not correct prior feature exploration or earlier project-wide searches;
all development conclusions remain exploratory. No independence or causal claim.

If both bounded breadth trials fail, do not add features or tune cutoffs merely
to chase these outer results. Diagnose the available inner/outer score and
cutoff-transfer evidence without fitting and identify the next evidence-supported
data/provenance requirement. Document what the available data do and do not
establish. An untouched cohort is still required for independent confirmation.

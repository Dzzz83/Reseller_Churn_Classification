# Exact baseline-catch ranking diagnosis — frozen before curve extraction

2026-10-17. Direct SKU, bagged SKU and latent mix fail operational criteria.
Before recommending another remedy, distinguish ranking at the exact baseline
catch count from cutoff slack. Use saved original/cadence/SKU/bagged/latent scores,
no fitting or operational cutoff selection. Common301/919/1201 groups only.

For each outer validation fold use the original baseline's SAVED operational TP
count as the descriptive recall target. For each procedure find the fewest FP
possible at score>=a tied-score boundary while catching at least that many TP.
Preserve ties; never split identical scores. Report target/achievedTP, FP and
actual TP/FP; compare candidate oracle burden with BASELINE oracle burden as well
as baseline actual burden. Otherwise baseline cutoff slack could masquerade as
ranking gain. If target0, define no-alert burden0. These are OUTCOME-DERIVED oracle
curve points, never deployable cutoffs or accepted performance. Do not save them
as operational threshold proposals. Report every fold, all adverse differences,
repeat sums and count of fold improvements/losses/ties, without inferential claims.

Also compute the same diagnostics separately within each inner validation fold
at the baseline saved pooled-inner cutoff. Each such score subset comes from
one fitted ensemble; do not treat pooled cross-model scores as one ranking. Inner
labels already selected the training cutoff, so inner metrics are descriptions,
not independent performance. Compare inner/outer patterns without causal claims
about training size or calibration. Repeats, snapshots and folds are dependent.
No October outcomes, changed labels/evaluation populations, fresh final evaluation
or automated remedy selection. Any next fitting protocol must be separately frozen.

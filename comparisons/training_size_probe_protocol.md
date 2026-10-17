# Fixed reseller-count sensitivity probe — frozen before fits

2026-10-17. Matched-rank burden shows better descriptive ranking for cadence/SKU
at original baseline catches, while operational criteria fail. This does not
establish calibration failure. A single strictly increasing mapping applied to
both inner and outer scores would preserve threshold-selected decisions; do not
fit a naive monotone calibrator as a supposed ranking fix.

Hypothesis to investigate, not assume: fewer training resellers weaken the SKU
procedure's tail ranking/score transfer. For each fixed outer fold, fit that same
unbagged direct-SKU procedure at50% and75% of outer-training resellers. Use a
single deterministic permutation of sorted training reseller IDs, seed2026;
fractions take nested prefixes, floor(N*fraction). Include ALL snapshots of each
selected reseller. Validation population/labels remain unchanged. Vocabulary,
weights and HGB are trained only on that subset, using the exact existing
support20/cap20/7-leaf/L2=1 procedure and5 original seeds. Require both classes;
report infeasibility/failure, never redraw. Reuse unchanged full-training scores.
150 new estimator fits; no fraction/seed grid beyond these two frozen probes.

This is a SCORE-SENSITIVITY diagnostic, NOT a fully nested candidate comparison:
reuse the already saved full-training SKU cutoff for all three sizes, explicitly
labeling the mismatch. Do not select a winner, promote a reduced-data procedure,
or claim fair operating performance from that fixed-cutoff probe. No new cutoff
selection or protected outcomes. Original baseline remains the operational
reference. Report raw score distributions, TP/FP at the reused cutoff, AP and
within-fold oracle FP at baseline actual TP (outcome-derived, never deployable),
all folds/repeats and vocabulary sizes. Save identities and selected reseller IDs.

Evidence supporting a larger-data hypothesis would be monotonically higher
within-fold mean AP and lower mean oracle FP with50→75→100% within EACH repeat.
Report every reversal; do not weaken that descriptive consistency check. It is
not an acceptance rule, uncertainty test or proof that collecting future data
will work. Training membership, sample size and learned vocabulary all change;
one nested subsample sequence does not isolate a causal sample-size effect.
Repeated/fold predictions are dependent. No bootstrap inference or operational
success claim is attached to this diagnostic. Any subsequent candidate needs a
separately frozen full nested protocol under the original acceptance standard.

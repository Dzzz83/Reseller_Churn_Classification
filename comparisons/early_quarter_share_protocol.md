# Earliest historical-quarter order share — cycle 8, frozen 2026-10-18

## Development observation and hypothesis

Seven continuation candidates failed complete acceptance. A no-fit history audit
reconciled all four historical quarter counts with annual/recent order counts for
every approved case. Existing baseline-space matching (unchanged k=1/3/5) found
persistent recent false alerts have lower earliest-quarter order share than caught
churners in all four snapshots for k=1 and k=5; k=3 has a small January reversal.
Most paired medians are zero and mean contrasts are small. Other quarter contrasts
reverse by period. Cadence-residual matching reverses the earliest-share direction
in April. Preserve all adverse results; do not call this stable seasonality or a
causal relationship. The feature was chosen after inspecting eight descriptive
count/share summaries: this is exploratory development adaptation.

Hypothesis: the oldest quarter's relative purchasing mass provides compact history
shape beyond original annual revenue and recent activity, helping separate recent
continuing buyers from fading longer histories. Test ONE addition to original HGB10,
not to cadence12 (whose residual evidence is weaker). No quarter/transform grid.
Prior annual-order count and basket breadth trials were different summaries and
failed; this feature is not credited with cadence's earlier gains.

## Fixed feature and estimator

For each approved snapshot t, count unique accepted orders in [t-12m,t-9m), divided
by unique accepted orders in [t-12m,t). Eligibility ensures a positive denominator.
Name: early_quarter_order_share. It is finite and in [0,1]. Use only saved accepted
pre-snapshot trajectories from persistent-error-investigation-20261016, verified
against its immutable manifest. No raw later exports, new source, history extension,
current attributes, outcome-derived quantity, or changed labels/cleaning.

Candidate = original ten features + this one share. Original HGB parameters,
31 leaves, L2=0, balanced FITTING weights, native missingness, five original seeds
and mean probabilities are unchanged. No tuning or preprocessing. Original HGB10
is the PRIMARY comparator, with unchanged cached fits/scores. 300 new fits.
StoreID and snapshot identify cases and never enter predictors. Historical
`discounted_line_share_12m` naming is preserved.

## Matched validation and common cutoff

Use all 1,375 cases, 212 churn cases, 422 resellers and the exact saved outer
301/919/1201 repeats x five folds; the same three inner grouped folds, seed78.
All snapshots of a reseller stay together at both boundaries. Fit weights and
models only within training. Every arm maximizes pooled inner-OOF precision subject
to recall strictly >.90; ties favor recall then higher cutoff, >= alerts, ties intact.
Outer labels select nothing. Natural validation populations stay unchanged.
This is mixed-period unseen-reseller evidence, not matured-label forecasting.

## Acceptance and prospective numerical-resolution safeguard

All previous practical/operating requirements remain: recall >= baseline and
strictly fewer FP EACH repeat; >=5% mean FP reduction; each repeat's snapshot,
activity, and snapshot/activity cell loses at most .05 recall and adds ZERO FP.
Undefined/missing/infeasible comparisons block acceptance. Margins are conservative
research defaults, not supplied business requirements.

Primary: 5,000 paired reseller-cluster draws, seed2026, carry snapshots/repeats
and both arms together. Continue cycle8 alpha spend .0000078125, divided across
two metrics and both tails: tail .000001953125. Require recall lower >=0 and
FP-reduction lower >0. If ANY primary gate fails, reject with no rescue. If all
pass, also require the existing 50,000-draw same-seed/tail confirmation.

ADDITIONAL safeguard, frozen before fits: a positive result remains provisional
until a fixed ceil(20/tail) = 10,240,000-draw simulation audit, seed2026, passes.
Use streamed multinomial reseller resampling and retain exact extreme order
statistics plus failure counts. For recall count draws with delta<0; for FP count
draws with reduction<=0. The one-sided Clopper-Pearson upper bound for each failure
probability, at Monte Carlo error tail, must be <= the frozen bootstrap tail.
This only adds a rejection gate; it never changes old failed decisions, margins,
cutoffs, tails, or permits a failed 5,000/50,000 gate to be rescued. Simulation
resolution is separate from statistical coverage: it does not fix adaptive search,
retraining/reselection, source uncertainty, or future population/calendar shifts.
The prospective alpha ledger is not a correction of all historical adaptation.

## Correctness, reporting, and independent verification

Before fitting: test calendar boundaries, positive denominator/range, duplicate/
future rejection, exact annual/recent reconciliation, no label inputs to feature
construction, and a synthetic nested fold with cached baseline parity. Independently
reconstruct all 1,375 feature values by a separate calendar-month algorithm.
Freeze input/code/config/runtime identities and splits before fitting. Preserve
all old outputs; use a new directory. No operational model change.

Save all training identities, inner/outer scores and selections, all repeat/fold/
period/activity/intersection TP/FP/FN/TN, recall, precision, alerts, undefined/sparse
flags, ranking/tied descriptive points, cutoff transfer, and all decisions. No-fit
verifiers reproduce feature/training hashes, group isolation, all cutoffs/counts,
cluster uncertainty and acceptance. A passed smoke establishes execution only.
Do not claim independently validated or final-test performance. No final evaluator,
model promotion, full-data deployment, push or merge is authorized.

Primary method context: Fader/Hardie/Lee's conditional purchase/dropout model
https://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf does not establish this
six-month label's global monotonicity. Historical purchase patterns can carry
information beyond summaries, but this feature's usefulness is an empirical
hypothesis, not inherited from BG/NBD. HGB mechanics:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html .
Exact binomial Monte Carlo bounds:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html .

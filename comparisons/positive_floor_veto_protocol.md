# Baseline alert with an OOF positive-floor veto — cycle 10

Frozen 2026-10-18 before constructing outer decisions. DEVELOPMENT ONLY.

## Observed weakness and structural hypothesis

The retained-logistic component reduces FP but loses recall; cycle9 cannot find
one component-score cutoff satisfying all inner recall and period-FP constraints.
Its binding FP lower bound is consistently April2013, while recall upper bounds
usually involve October2012/January2013. A single cutoff on that component is
therefore infeasible under the frozen guards. The fixed mean blend also failed.

Hypothesis: preserve the original baseline alert gate, and use the calibrated
component only as a conservative veto of its lowest-risk alerts. This is a new
nonlinear decision composition, not a relaxed guard, new blend weight, or threshold
chosen from outer errors. The AND rule CANNOT add false alerts or new catches on
any population. To meet the objective it must remove enough baseline false alerts
while losing ZERO baseline-caught churners in every development repeat.

## Fixed training-only procedure and common cutoff objective

Components are the unchanged original HGB10 and fixed retained_logistic from
cycle1 (cadence12, symmetric-half classifier/reference maps). Reuse their exact
saved inner/outer scores and identities; no new model/calibrator fits or loading.
All inputs, seeds, preprocessing, weights and grouped boundaries remain frozen.

Within each outer-training partition:
1. Select the original baseline reference cutoff from its innerOOF scores by
   precision maximum with recall strictly>.90, ties recall then higher cutoff.
2. Identify INNER cases that baseline alerts and whose training label is churn.
   Let veto_floor be the MINIMUM retained-logistic OOF score among these cases.
   No quantile, scaling factor, margin or floor grid is searched. Equality with
   the floor is retained; only component scores STRICTLY BELOW it are vetoed.
3. Define the candidate score as baseline_score when component_score>=veto_floor,
   and 0 otherwise. It is a decision score, not a calibrated probability.
4. For BOTH baseline and transformed candidate use ONE common cutoff objective:
   maximize innerOOF precision, ties recall then higher cutoff, subject to
   recall>.90, TP>=baseline-referenceTP and cutoff>=baseline-reference cutoff.
   The baseline cutoff must reproduce exactly. Because the floor retains every
   baseline-caught INNER churner, the candidate cutoff must also equal the
   reference cutoff; assert this rather than retuning. Score ties remain intact.
5. Apply that learned floor and unchanged baseline cutoff to outer scores. If the
   baseline reference cutoff is0, bounded-score suppression is unavailable, so
   explicitly fall back to baseline for the entire fold. If there are no caught
   inner churners, the procedure is infeasible; never use a favorable subset.

No outer labels/scores choose the floor, cutoff, component, margin or direction.
Freeze the floor before reading each saved outer prediction file. Every outer
case remains in the natural evaluation population. Vetoing an alert is not removing
an account. The outer alert set must be a subset of baseline in EVERY fold/group.
Recall transfer is NOT guaranteed by the empirical inner floor: losing even one
baseline catch fails the corresponding outer repeat requirement. OOF models and
repeated histories are dependent, so no NP/conformal coverage theorem is claimed.

## Population, acceptance and uncertainty

Exactly1375approved cases/212churners/422resellers;four prescribed snapshots,
301/919/1201 x5outer,three inner grouped folds seed78. Reseller snapshots stay
together in fitting/reference/validation. Mixed-period grouped estimand only.
The feature schema is the fixed union of original10 and cadence12,with IDs/date
metadata excluded. No data/source/window/label/cleaning change.

All outer acceptance remains unchanged: recall>=baseline ANDFP<baseline EACH
repeat;>=5%meanFP reduction;each repeat's snapshot/activity/intersection loses at
most.05recall and addsZEROFP;undefined/missing/infeasible comparisons block. The
subset property alone is NOT success. These margins remain research defaults.

Primary5000paired reseller-cluster draws seed2026,all snapshots/repeats carried
jointly. Continue cycle10 alpha spend.000001953125,tail.00000048828125 across two
metrics/both tails. Require recalllower>=0,FPreductionlower>0. If ANY primary gate
fails,reject with no rescue. If all pass,require same-seed/tail50000confirmation.

Additional numerical verification for THIS prospectively frozen procedure:
require EXACT empirical reseller-bootstrap sign-failure probabilities strictly
below the same tail. Integer polynomial convolution counts all422**422 ordered
cluster resamples without Monte Carlo or floating-tail approximation. Check that
resampled churn denominators are identical across repeats,so mean recall's sign
is the sign of summed repeat TP changes. Treat undefined-denominator mass as
adverse. FP failure is summed reduction<=0;recall failure is summed TP change<0.
Use exact integer cross-products against the decimal-rational tail. Independently
verify the veto special case: FP deltas are nonnegative,so no-improvement
probability=(number of zero-benefit resellers/422)**422. If no TP is lost,recall
negative probability is0;undefined mass remains disclosed/conservative.

This deterministic verification strengthens the same conditional sign gates and
avoids unresolvable tiny simulated tails. It is prospective for cycle10;the frozen
cycle8/9 additional simulation plans and their failed decisions are untouched.
Do not rescue failed primary/confirmation gates using exact arithmetic. Exact
counting is only exact for the empirical fitted cluster bootstrap,not statistical
coverage under adaptive search,refitting,calendar dependence or future shift.
Report existing approximate interval magnitudes alongside exact sign probabilities.

## Correctness, verification and reporting

Test exact ties at the veto floor, baseline cutoff reproduction, training-catch
preservation, zero-cutoff fallback, score/case alignment, and the alert-subset
invariant. Exhaustively verify integer convolution against every small synthetic
resample, and direct recall/FP statistics. Independently reconstruct every learned
floor/outer score/decision,all component identities,counts and complete acceptance.
For a provisional winner,independently verify closed-form sign probabilities and
all frozen stages before any DEVELOPMENT IMPROVEMENT claim.

Save source/code/protocol/runtime/split identities before construction,each floor
and selected training case keys,all paired inner/outer scores,cutoffs,all repeats/
folds/periods/activity/intersections,undefined/sparse flags,ranking/tied descriptive
points,cutoff transfer and exact probabilities. No discarded folds or cherry-picked
periods. No full-data model promotion,protected evaluation,push,merge or deployment.

Primary method context:training-only threshold selection
https://scikit-learn.org/stable/modules/classification_threshold.html ;
Tong/Feng/Li(2018),https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/ discusses
order-statistic error control under independent held-out samples. This OOF floor
DOES NOT satisfy/claim that paper's independence-based guarantee. Exact bootstrap
calculation context:Clark/Warr(2021),https://arxiv.org/abs/2107.08019 and
Bertsimas/Sturt(2017),https://optimization-online.org/wp-content/uploads/2017/09/6211.pdf .
No novelty or inherited churn-performance claim is made.

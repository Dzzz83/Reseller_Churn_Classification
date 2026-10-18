# Period-weighted product engagement — frozen development comparison, 2026-10-18

## Motivation, previous failures and bounded scope

User asks for information that moves churners above continuing buyers without
many more false alerts. The DEVELOPMENT-only hard-recent warning audit finds
lower annual distinct-product counts in missed cases vs ten-feature-matched
continuing buyers at k3/k5 in all four periods, attenuating to ties at k1.
Simple revenue/cadence warning flags overlap heavily. Annual breadth previously
added ~.0066 mean within-fold recent AP under ordinary balanced HGB but failed
full operational acceptance. SKU/PCA representations did not beat their cadence
control's recent AP, so do not repeat them or claim their comparison with original
HGB isolates product information. All historical failures remain failures.

Two predeclared arms, no search after results:
A period_product_breadth: original10 + unique_products_12m.
B period_product_density: original10 + unique_products_per_order_12m, defined as
annual DISTINCT ProductIDs / annual accepted-order count. This is annual breadth
per order exposure, not mean basket size. It distinguishes product variety from
simply having more opportunities to purchase. No logs, caps, counts combined,
projection, product selection or parameter alternatives. Both are outcome-free
row-local features from accepted purchases in[t-12m,t). Verify all1375counts from
saved annual SKU profiles and historical order trajectories against saved history.
No current profile, cost, service, status, future purchases or extra history.

The repeated breadth hypothesis is meaningful only because the fitting target
now uses fixed equal period/class weights instead of the earlier ordinary class
weights. Do not claim novelty or that the weight/feature interaction is established.
A controlled null result is useful and will not authorize an arbitrary third arm.

## Fixed model, inputs and isolation

Primary comparator is the already chosen period/class HGB10. Use unchanged
cached fitting scores from results/period-weighting-20261018; identical approved
1375cases/212churnlabels/422resellers/four snapshots. Exact saved301/919/1201 x5outer
partitions and three inner grouped folds seed78. Every reseller's snapshots stay
together at every boundary. IDs/date are not predictors. Natural evaluation rows
and labels unchanged. Historical discounted_line_share_12m schema retained.

All arms originalHGB100iterations,learning_rate.1,31leaves,min_leaf20,L2=0,
early_stoppingFalse,native missingness,average seeds42,78,88,1034,2026. Fitting
weights N/(8*n_snapshot,class),training labels/cells only. No resampling,
calibration or tuning. Each new arm300HGBfits. Cache reference identities and
verify fitting hashes. Persist every inner/outer prediction, fitted model,
weight vector,train hash,parameter and cutoff before assessment.

## Common cutoff and aggregation, frozen before fitting

PRIMARY operating policy for ALL arms: maximize pooled INNER-OOF recall subject
to precision >=40%,ties higher precision then higher cutoff. Intact ties,>=alerts.
This is a NEW controlled comparison under the user's precision policy, not a
reinterpretation of old recall>.90 acceptance. Comparator is fixed period/class10,
not a substituted ordinary-class reference. No outer label tunes a threshold.
No feasible cutoff =>record explicit infeasible fold; save scores/ranking,block
operational acceptance. No fallback or customer removal. Undefined/missing metrics
block the corresponding gate. Original10 scores may be saved for context only.

Pool outer held cases once per repeat for operating counts. Report all15folds,
3repeats,4snapshots,activity groups and intersections,undefined/small flags. Within
fold recent-purchase AP is the primary ranking measure; average five folds within
repeat then three repeats. Recent ROC-AUC/supporting pooled ranks reported too.
Do not compare mixed-model scores as if they came from one fitted scorer.

## Success defaults and uncertainty

Retain earlier practical research margins; these are not company requirements.
Require TP>=reference and FP<reference EVERYrepeat;at least5%meanFP reduction;
eachrepeat snapshot/activity/intersection recall loss<=.05 and ZEROaddedFP.
Also require mean within-fold recent AP gain>=.01,nonnegative gain in eachrepeat,
and nonnegative mean recent ROC-AUC gain. Better average/ranking alone is NOT
operational success. All original development acceptance failures remain unchanged.

Paired reseller-cluster bootstrap5000draws seed2026 carries all snapshots/repeats
jointly. Require lower recall difference>=0 and lowerFP reduction>0. Continue the
existing alpha-spend ledger: armsA/B use cycles11/12 respectively,spend.001/2**(k-1),
tail spend/4 across two operating metrics/both tails. Ranking point gates add
requirements but no familywise ranking-confidence claim. Report descriptive
95%paired cluster ranking intervals separately; they omit adaptive research,
retraining,calendar dependence and future shifts. New fold seeds are not new data.

Only if every primary point/ranking/subgroup/5k gate passes,require50000draw
same-seed/tail operating confirmation,then exact empirical bootstrap sign checks
(integer convolution),with undefined denominator mass adverse. No failed gate
receives extra-draw rescue. Exactness is numerical for the fitted empirical
bootstrap,not a new independent or adaptive-search guarantee. Tiny simulated
tails remain underresolved; exact signs strengthen a prospective winner only.
No automatic October evaluation under this research task.

## Provenance, verification and reporting

Frozen source/protocol/input/runtime identities before fitting. Verify model,
feature and weight identities and all cutoffs/counts independently without refits.
Synthetic checks:feature formula/zero-order rejection,missing/future snapshots,
training-only weights/held-label exclusion,model reload scores,threshold ties and
ranking-bootstrap weighted tie handling. Retain interrupted outputs without
rerunning unchanged completed fits. Report both planned candidates,including
failures and rank/operating disagreement. Journal and standalone checkpoint at
cycle completion. No operational modification,push,merge or deployment.

Method context:scikit-learn leakage guidance requires fitting transforms only on
training data: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage .
PCA documentation informed reviewing prior projections,not this count-only trial:
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html .

Attribution limit:the density ratio introduces annual order exposure as well as
product breadth. If it helps,do not attribute its gain uniquely to product variety;
a separate controlled annual-order-count comparison would be needed. This study
predeclares only the two features above and will not append that arm after results.

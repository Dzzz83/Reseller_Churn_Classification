# Fixed activity-routed experts — frozen 2026-10-18, contingent cycle 4

Execute only if reseller-weighting cycle3 fails complete acceptance. Baseline
development diagnoses show recent-purchase recall around84% with many false
alerts,versus100% recall and100%FPR without recent purchase. Calibration fixes
average transfer but not period ranking. Hypothesis: distinct fixed experts may
avoid interference between these behavioral regimes,using no new features or
outcome-based subgroup discovery. No specialist protocol was located among the
reviewed prior experiment protocols. This does not prove universal novelty.

Route by n_orders_3m>0,known before prediction. Fit BOTH groups with original
HGB10 architecture/parameters31leaves,L2=0,originalfive seeds/native missing.
Compute original balanced weights on the ENTIRE fitting partition,then supply
the appropriate rows/weights to each expert WITHOUT rebalancing within groups.
This keeps every training row,original row weight,feature and label;only the
model responsible for each row changes. Every held row routes to its matching
expert. No customer is discarded and natural evaluation is unchanged. Both
classes required in each fitting route;missing class is infeasible,not a fallback
model. Fit each group even if a held partition has no such rows;record empty
held route explicitly.600new expert fits,300cached original baseline identities.

The route is fixed,not learned or tuned;no alternative recency thresholds,
activity-specific cutoffs,extra cadence/price/SKU features,calibration,parameter
search or class-prior correction. Both procedures select one common-policy cutoff
per outer model using all natural innerOOF cases:precision maximum with recall>.90,
ties recall then higher cutoff,>=alert. Outer labels do not select routing,model
or cutoff. Five seeds may be identical;report spread rather than claiming extra
independence. Same1375/212/422 cases and301/919/1201 x5outer/3inner memberships,
reseller-disjoint boundaries. Grouped estimand,not chronological forecasting.

Acceptance unchanged: recall>=baseline andFP<baseline EACH repeat,>=5%meanFP
reduction,<=.05recall loss andZEROaddedFP each repeat's snapshot/activity/intersection,
undefined recall/missing/failed folds block.5000paired reseller-cluster draws
seed2026 carry all snapshots/repeats;cycle4 spends.000125 -> two metric/both-tail
probability.00003125. Require recall lower>=0,FP-reduction lower>0;50000same-seed/
tail confirmation only if ALLprimary checks pass. Defaults,extreme-tail imprecision,
adaptive repeated development,retraining/calendar exclusions remain disclosed.
No post-result gate/threshold/architecture changes or promotion.

Save entire fitting-population and routed-subset hashes,exact weights,seeds,
parameters,routed held keys/scores,component spreads,all inner/outer scores,
cutoffs,operating and ranking tables,transfer counts,input/code/runtime snapshots
before fitting/assessment. Independent no-fit verification checks all600subset/
weight identities,full routing coverage,scores,30cuts,540operating rows,5000draws
and full decision. Synthetic smoke covers both routes,empty held route,missing
training class rejection and no held labels. All reports include adverse periods.

Primary research context:Jacobs,Jordan,Nowlan,Hinton(1991),Adaptive Mixtures of
Local Experts,https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf .
The paper discusses specialization for known subtasks;this deterministic two-HGB
route does not reproduce its learned neural gating method or inherit its results.
Original HGB mechanics:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html .
No protected-test evidence selected this method;no test/source-outcome access,
full-data fitting,model deployment,push or merge.

## Execution amendment before any real predictions — version 2

Initial results/activity-experts-20261018 failed during the FIRST inactive expert's
binning;no real expert fit completed,no held scores or candidate outcomes were
produced. Synthetic60row reproduction confirms sklearn1.9.1 _find_binning_thresholds
calls sliding_window_view on zero observed values for an all-NaN column,raising
ValueError. Inactive histories naturally lack weighted_discount_3m observations.
Preserve the interrupted code/protocol directory. No estimator monkeypatch/library
change or invented observed value is permitted.

Revised implementation rule,frozen before restart:each expert omits ONLY columns
that are entirely missing in ITS FITTING rows. Record actual model_features and
omitted_all_missing;retain all10 fields in input/training hashes. There is no
outcome/validation-based feature selection,and no observed information is removed.
Other missing values remain native. Constant observed columns are retained.
This necessary all-missing-column omission makes the executable procedure explicit;
it is not a new feature-usefulness claim. Tests cover this failure and the omission.
Restart results/activity-experts-20261018-02 with unchanged parameters,seeds,
weights,partitions,cutoff objective,acceptance and alpha ledger. Save both versions.

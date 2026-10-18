# Inverse reseller-history weighting — frozen 2026-10-18, cycle 3

Observed DEVELOPMENT structure:38 resellers have1 snapshot,51 have2,97 have3,
236 have4. Row churn rates by observed count:100%,85.29%,11.34%,5.72%.
These are study-presence associations,partly reflecting eligibility/survival,
not causal evidence or prediction features. Row-balanced training gives each
row equal within-class influence,not each reseller. Earlier unit-weight trials
and reseller-bootstrap product fits did NOT test inverse-history class weighting.
Hypothesis: reducing repeated histories' relative fitting influence within each
class may improve natural-row high-recall decisions. It can also worsen them;
no claim equal-account fitting estimates the same target as equal-row fitting.

One controlled candidate: ORIGINAL ten-feature HGB with original31leaves,L2=0,
other parameters unchanged,5seeds42/78/88/1034/2026,native missingness. For each
FITTING partition only,let n_i be that reseller's number of fitting rows;v_j=1/n_i.
For y in{0,1},set w_j=v_j*N/(2*sum_{k:y_k=y_j}v_k). Thus weights sumN and each
class totalN/2,matching original balanced-class total scale,while relative weights
within class change. Counts/labels from held inner/outer rows NEVER affect weights.
StoreID/history count remain metadata,not predictors. No oversampling,synthetic
histories,weight exponent/cap/search,feature/label/cleaning/window change. A reseller
with both labels does not necessarily have equal TOTAL final class-balanced weight;
report that nuance. Weighting is an experimental loss choice,not selection-bias cure.

Only approved1375rows/212churners/422resellers. Reuse cadence-stability original
baseline/outer and3-fold inner memberships. Repeats301/919/1201 x5 outer folds.
Both procedures select precision-max inner-OOF cutoff with recall>.90,ties recall
then higher cutoff,>=alerts,intact ties. Same natural validation population,
reseller isolation at both levels,no outer-label decisions.300newHGB fits,
300cached baseline identities. No full-data model or protected evaluation.

Acceptance UNCHANGED from continuation cycles: recall>=baseline,FP<baseline
EACH repeat;>=5%meanFP reduction;each repeat's snapshot/activity/intersection
<=.05recall loss,ZERO addedFP;undefined recall/missing/infeasible folds block.
Paired5000reseller-cluster draws seed2026 carry every snapshot/repeat together.
Cycle3 spends.00025 of prospective ledger -> two endpoints/both tails .0000625.
Require recall lower>=0,FP-reduction lower>0;50000draw same-seed/tail confirmation
ONLY if all primary gates pass. Defaults are conservative research choices.
Extreme-tail/bootstrap/adaptive research/calendar and retraining limits remain.
No relaxed margin,alternate seed,more draws or retuning can rescue a failure.

Save each weight vector/training hash,seeds,model parameters,inner/outer scores,
cutoffs,all operating and ranking tables,transfer sizes,input/code/runtime hashes.
Focused synthetic checks require within-fitting count calculation,equal class
weight totals,scaleN,one-row-per-account equivalence to ordinary balanced weights,
and independent no-fit reconstruction of all300weight identities,cutoffs/counts/
bootstrap/safeguards. Preserve historical outputs and operational code. This
controlled trial holds features/architecture fixed and changes only training weights.

Primary sources: Williamson,Datta,Satten(2003),Marginal Analyses of Clustered Data
When Cluster Size Is Informative,https://onlinelibrary.wiley.com/doi/10.1111/1541-0420.00005
motivates distinguishing cluster-weighted and row-weighted targets; its marginal
inference results do NOT establish HGB performance. sklearn HGB sample_weight docs:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html .

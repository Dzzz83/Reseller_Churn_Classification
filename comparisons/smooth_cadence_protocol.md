# Smooth additive cadence learner — frozen 2026-10-18, cycle 2

Development hypothesis: retained logistic calibration improves average score
transfer but loses recall and period safeguards. At exact baseline catches,
its descriptive tied-score burden worsens in October2012 in two repeats and
April2013 in one. A score/cutoff fix alone is insufficient. Test one smooth
additive decision function as a different inductive bias from HGB partitions.
The prior linear10 comparison failed; it had no smooth nonlinear terms or cadence
inputs. This is not a rerun of it or an assertion that all tree errors are variance.

Use original ten + overdue_ratio + gap_cv from verified cadence-stability data;
no new sources/history/labels. Historical feature naming preserved. Pipeline:
training median imputation (keep_empty_features=True); cubic B-spline expansion
of each numeric feature,5 quantile knots,include_bias=False,extrapolation=constant;
concatenate missing indicators for all12 features; standardize all resulting
columns using training-only mean/std; LogisticRegression(C=1,solver=lbfgs,
max_iter=2000,tol=1e-8), balanced sample weights derived from FITTING labels only.
All preprocessing unweighted, training-only. No interactions, knot/C/degree
search, feature selection or resampling. Constant/extrapolated and missing inputs
must yield finite scores. One deterministic fit per partition:60 fits.

Primary original balanced HGB10 comparator and cached scores/inner memberships
unchanged: repeats301/919/1201,5 outer,3 inner grouped folds seed78. SAME common
cutoff objective precision maximum subject pooled inner-OOF recall>.90; tie recall
then higher cutoff; >= alerts, intact ties. Outer labels select nothing. Preserve
all reseller snapshots at every boundary, natural validation rows,1375/212/422.
Grouped mixed-period estimand is not historical forecasting. Repeated development
adaptation and prior inspected-test history preclude independent confirmation.

Acceptance exactly as cycle1: recall>=baseline and strictly fewer FP EACH repeat;
>=5% mean FP reduction; period AND activity AND snapshot/activity intersections
EACH repeat <=.05 recall loss,ZERO added FP; undefined recall or infeasible fold
blocks acceptance. 5000 paired reseller-cluster draws seed2026 move every snapshot/
repeat together. Cycle2 spends .0005 across two metrics/both tails -> tail .000125,
continuing the cycle1 prospective alpha ledger, not correcting all earlier searches.
Require recall lower>=0,FP-reduction lower>0;50000 same-seed/tail confirmation ONLY
if every primary check passes. Defaults are research choices, not business limits.
All planned fits and failures reported; no reduced favorable population pooling.

Save training hashes, imputation medians, all spline knots/coefficients,scaler
statistics,logistic coefficients,inner/outer scores,cutoffs,code/input/runtime
identities BEFORE assessment. Independent verification reconstructs the spline
basis from saved knots via scipy BSpline, checks medians/scaling/fit identities,
held scores,cutoffs,all operating counts and decisions without refitting. Supporting
ranking/tied oracle points remain descriptive. Report every fold/repeat/period/
activity/intersection,undefined flags,cutoff transfer,component sizes. No changes to
operational pipeline,protected evaluation,promotion,push or merge.

Method sources: official sklearn SplineTransformer documentation
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html
and Stanford Statistical Learning GAM lecture by Hastie/Tibshirani/Taylor
https://www.youtube.com/watch?v=3aMB51GMUyQ . Smooth additive effects can represent
nonlinear associations; these sources do not establish churn performance here.

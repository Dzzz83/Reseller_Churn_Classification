# Fixed training-weight ablation — frozen before fitting

2026-10-16. The fixed linear procedure failed, adding 302.67 FP per repeat on
average and losing recall in a repeat. Nonlinear separation remains necessary
for the methods tried; this is not proof of impossibility for all linear features.
No coefficient, cutoff or blend is retuned afterward.

Hypothesis: the arbitrary 1:1 weighted class target may change finite-capacity
tree partitions/ranking unfavorably. The high-recall requirement is already
selected inside training; balanced fitting need not be optimal for ranking near
that point. This is a testable hypothesis, not a diagnosed cause of current errors.
The matched comparison changes fitting weights, not only the probability cutoff.

Previous imbalance results are adverse/mixed evidence: seven-feature temporal
HGB AP is 0.2915 unweighted versus 0.3091 balanced; forest AP is 0.3374 versus 0.3175.
Those use two temporal folds and a 0.5 diagnostic threshold, not the present
nested grouped ten-feature objective. Do not omit those historical findings or
claim training weighting has never been studied.

Freeze two unit-weight procedures, with no weight grid or new features:
1. Original ten-feature HGB, original 31-leaf/L2=0 settings, five original model
   seeds, native missing handling, **no balanced sample weights**.
2. Original ten-feature 300-tree forest, same previously frozen parameters,
   training-only median+indicators and five seeds, **no balanced sample weights**.
Class_weight remains None for the forest; do not replace external weights with
another implicit class weighting. No resampling or calibration.

Primary comparator for BOTH remains the unchanged balanced original HGB on the
same 301/919/1201 outer/inner partitions and 1,375 natural cases. Reuse verified
balanced HGB and balanced forest fits/scores. The latter is a separately labelled
reference for the within-architecture weight contrast, not a weakened operational
baseline. Fit only the two new unit-weight procedures: 600 estimator fits.

All cutoffs use inner OOF precision maximization with recall strictly >90%; ties
recall then higher cutoff. Training preprocessing and all snapshots of a reseller
stay isolated; validation labels never choose parameters/weights/cutoffs.

Require recall>=original baseline and FP<baseline in each repeat, >=5% mean FP
reduction and <=5-point mean recall loss for each activity/snapshot slice. These
margins remain research defaults and are unchanged. Use 5,000 paired reseller-
cluster draws, seed 2026, carrying snapshots/repeats together. Recall lower bound
must be>=0 and FP-reduction lower bound>0 at tail 0.05/(22*2), including the two
new procedures after the previously counted 20. This does not correct the whole
adaptive project history, refitting/reselection or population shift; label all
findings exploratory. Report both primary comparisons and secondary weight
contrasts, every fold/subgroup, uncertainty, high-recall ranking and failures.
No original 0.06 baseline policy, October outcomes, new feature family, protected
final evaluation or deployment enters this trial.

If a candidate passes every primary check with 5,000 draws, confirm its conditional
lower bounds with 50,000 draws at the same seed and family tail before acceptance.
This is a predeclared Monte Carlo precision safeguard for the extreme tail, not
new independent evidence or a search over bootstrap seeds. Both lower bounds
must still meet the same signs. Save the larger-draw result and any failure.

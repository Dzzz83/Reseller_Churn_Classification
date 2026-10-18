# Retained component logistic calibration — frozen 2026-10-18

Observed DEVELOPMENT weakness: cadence ranking improves but inner-to-outer
operating rates differ and April false alerts rise. Prior raw symmetric-half and
unsupervised reference-percentile procedures both failed acceptance. Hypothesis:
label-informed, regularized sigmoid maps fitted to each component's disjoint
reference resellers, retained with that component, may stabilize scores before
averaging. A common monotone map cannot improve one model's ranking; different
component maps before averaging can change ranking. This is a new supervised
ensemble procedure, not proof calibration was the cause.

Sources: scikit-learn calibration guide
https://scikit-learn.org/stable/modules/calibration.html describes retaining
classifier/calibrator pairs and disjoint fitting/calibration samples. Niculescu-
Mizil and Caruana, Obtaining Calibrated Probabilities from Boosting (2005),
https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf motivates
investigating distorted scores, not a claim its AdaBoost findings apply to HGB.
Small/dependent reference samples remain a limitation. Use parametric calibration,
not an isotonic search.

## Fixed procedure and population

Reuse ONLY results/crossfit-percentile-20261017's saved DEVELOPMENT component
scores, fitting/reference keys, hashes and grouped partitions. Validate all 600
component identities against development_input.csv before calibration. No HGB
refits, model deserialization, new source exports, history extensions or test
access. Approved 1375 rows/212 churners/422 resellers, four snapshots only.
Historical discounted_line_share_12m is retained; the operational rename changes
no values. Cadence12 uses the original ten plus overdue_ratio and gap_cv,
seven HGB leaves/L2=1; all other parameters, native missingness, balanced FITTING
weights and five seeds unchanged. Two stratified reseller halves (seed42) supply
complementary fitting and reference samples inside every inner/outer partition.
Every reseller remains together at all levels. Calibrator labels are from the
reference subset ONLY; held inner/outer labels are unavailable to calibration.

For every component, fit sklearn LogisticRegression(C=1,solver='lbfgs',
max_iter=1000,tol=1e-8,fit_intercept=True), no class/sample weighting, on
log(p/(1-p)), clipping p to [1e-12,1-1e-12]. This is regularized logistic score
calibration, not an assertion of exact Platt smoothing. Retain slope/intercept,
reference labels/scores/keys, mapped held scores and convergence status. Negative
slopes are reported, never silently reversed. Average all 10 mapped scores
(two halves x five seeds); identical seeds add no independent evidence. No
calibrator/model setting, clipping, half count, blend weight or feature tuning.

Primary comparator: original balanced HGB10 cached scores. Secondary raw-half
control is preserved and reported, not another chance to replace the baseline.
Outer repeats301/919/1201 x5 folds; same cached three inner folds seed78.
ALL procedures independently maximize pooled inner-OOF precision with recall
strictly >.90; ties recall then higher cutoff; scores>=cutoff alert. Cutoffs use
intact ties, never outer labels. Every arm has identical natural evaluation keys,
labels and partitions. This grouped estimand is unseen reseller identities across
mixed periods, not matured-label forecasting. No new confirmation cohort claimed.

## Acceptance frozen before fitting

Require ALL: baseline recall preserved and strictly fewer FP EACH repeat; >=5%
mean FP reduction; paired reseller-cluster recall lower>=0 and FP reduction
lower>0. Period safeguards EACH repeat: <=.05 recall loss and ZERO additional FP
in each snapshot and activity group. Also apply the same limits to snapshot/activity
intersections; undefined recall blocks acceptance. All folds remain diagnostic,
with infeasible/missing predictions blocking acceptance, never favorable-subset
pooling. Margins are conservative research defaults, not business requirements.
No threshold or criteria changes after outcomes. Both-class reference requirement
and any convergence failure make the candidate infeasible; preserve failure.

Use5000 paired reseller-cluster bootstrap draws seed2026; carry all snapshots,
repeats and arms together. New continuation alpha-spending ledger: cycle1 spends
.001 across two metrics and both tails -> tail .00025. Subsequent justified
cycles spend .001/2**(cycle-1), total<=.002; no reset per candidate. This is a
conservative prospective continuation budget, NOT an exact correction of earlier
adaptive searches. If all primary checks pass, require same signs at50000 draws,
same seed/tail. Extra draws never rescue a failed primary. Intervals omit
retraining/reselection, adaptive hypothesis choice and calendar/population shift.
Extreme-tail estimates with5000 draws are imprecise; disclose this explicitly.

Report TP/FP/FN/TN,recall,precision,alerts,n/undefined flags for all repeats/folds,
periods,activities/intersections; supporting ranking with ties and descriptive
outcome-derived operating points. Report cutoff distribution and unchanged-cutoff
inner/outer rates, actual component training/reference sizes, raw control changes,
all convergence and slope signs. Independent verifier reconstructs isolation,
hashes, logistic score formulas, means, cutoffs, confusion counts, safeguards and
bootstrap decisions without classifier/calibrator fitting. Save input/code/runtime
hashes before calibration; new directories only. No operational pipeline changes,
promotion,push,merge or final evaluation. A failed comparison motivates diagnosis,
not calibrator parameter search.

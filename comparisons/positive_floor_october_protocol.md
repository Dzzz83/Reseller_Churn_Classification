# Authorized positive-floor veto evaluation — 2026-10-18

User authorization: “evaluate on the final test set.” This evaluates the exact
positive_floor_veto candidate just discussed. Its development acceptance FAILED;
this evaluation does not change that decision. October was previously inspected,
so this is descriptive reuse of an inspected test cohort, not independent testing.
No alternative candidate, margin, threshold grid or test-based revision is allowed.

## Frozen preparation, before reading supplied outcomes

Use all 1,375 approved development rows (212 churn cases, 422 resellers, snapshots
2012-07-01, 2012-10-01, 2013-01-01, 2013-04-01). Six-month labels have matured by
2013-10-01. Reuse the verified original baseline's three-fold grouped OOF scores
(seed78) and full-development baseline scores from the prior checkpointed
unit-revenue October preparation. Verify exact development cases/features and
baseline specification, seeds, training identities and model scores. No use of
its candidate model or outcome report. Historical feature names are preserved.

Train the unchanged retained-logistic component in each of those three inner
partitions, and once on all development. Within a partition use two stratified
reseller halves (seed42), original cadence12 (ten original features plus
 overdue_ratio and gap_cv), HGB
seven leaves/L2=1 and remaining original settings, balanced fitting-only weights,
native missingness, five seeds42,78,88,1034,2026. Fit each logistic map only on its
disjoint reference half: C1/lbfgs/max_iter1000/tol1e-8, logit clipped1e-12,
no weights. Average all ten retained classifier/calibrator predictions.
No held labels enter fitting; all reseller snapshots stay together at each level.

On pooled development OOF, select the original baseline cutoff by precision
maximum subject recall strictly>.90, ties recall then higher cutoff. Learn veto
floor as MINIMUM component OOF score among baseline-caught churners. Keep equality;
suppress baseline score to0 only below that floor. Common constrained cutoff
objective and baseline-catch preservation must reproduce the reference cutoff,
as in comparisons/positive_floor_veto.py. Zero-cutoff whole-procedure fallback is
unchanged. No cutoff/floor is learned from October. Save protected-inner keys,
OOF predictions, learned rule, fitting/reference identities, retained estimators,
runtime/code/input identities and unlabeled October predictions before outcomes.

## Separate outcome stage

Only after verifying the preparation manifest and prediction checkpoint, join
supplied inspected outcomes in results/current-model-diagnosis/october_cases.csv,
verified against that source manifest. No source queries or label construction.
All492 October resellers remain. Compare the veto with original10 under its SAME
development cutoff policy (primary), and saved historical0.06 rule (secondary).
Report TP/FP/FN/TN, recall, precision, alerts, AP/AUC, recent/no-recent groups and
paired decisions. Report fixed5000reseller bootstrap draws seed2026,95%percentile
intervals as descriptive conditional uncertainty; no new hypothesis-test claim.
Undefined rates flagged. No test tuning or deployment. Development failure stays.

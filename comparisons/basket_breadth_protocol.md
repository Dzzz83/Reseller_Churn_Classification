# Last-order basket breadth — frozen cycle 2 protocol

Frozen 2026-10-16 after the matched-history exploration and before candidate fits.
Exploratory development reuse throughout; no independent confirmation claim.

## Hypothesis and methods

The ten-feature baseline omits basket breadth. Persistent missed churners tend
to have fewer last-order lines than similar correctly unalerted non-churners.
Test one candidate: original ten features plus `last_order_lines`, the accepted
line count of the latest order in [t-12 months,t), ordered by OrderDate then
SalesOrderID. Main feature window, eligibility and labels remain unchanged.
All cases have exactly one last-day order in the reconciled source. No tuning,
feature search, additional regularization, calibration, resampling or imputation.

Baseline and candidate use identical frozen HistGradientBoosting parameters,
balanced weights computed on training labels, native missing handling, and mean
probabilities over seeds 42,78,88,1034,2026. Both are fitted freshly in the same
current runtime, so runtime changes cannot affect only one arm. Historical saved
baseline scores are a fidelity reference, not a weakened comparator.

Use the same sorted 1,375 rows, 212 labels and 422 reseller identities; read
`results/repeated-grouped-precision/development_input.csv` with round-trip floats.
Merge the independently reconciled last-order count one-to-one. Use the 15 saved
outer partitions (seeds 42,78,88; 5 folds) and verify group isolation. Within each
outer training partition use the established three StratifiedGroupKFold inner
splits with seed 78 and all reseller snapshots together. Both arms maximize
inner OOF precision subject to recall strictly >90%; ties favor recall, then
higher cutoff. Preserve all score ties; predict score >= cutoff. Outer outcomes
never choose thresholds. No full-data model or operational cutoff is produced.

## Acceptance, frozen before fitting

Primary calculation: pool the five held-out folds once per repetition, compute
TP/FP/FN/TN separately for each arm, then pair within repetition. Every repetition
must have candidate recall >= baseline recall and candidate FP < baseline FP.
Require at least **5% fewer FP on average across repetitions**. Five percent is
a research default (about 26 of the historical 521 false alerts), not a supplied
business requirement. It avoids treating one or two avoided alerts as practical
success. Report every individual fold; no fold-by-fold noninferiority claim.

Stronger evidence: paired 5,000-draw reseller-cluster bootstrap, seed 2026,
carrying all snapshots and repetitions for a reseller together. Require the
2.5th percentile of mean recall difference >=0 and the 2.5th percentile of mean
FP reduction >0 (joint one-sided 95% via Bonferroni for two endpoints). Report
both full intervals. This is conditional on fitted models/splits/cutoffs and
omits retraining/reselection, adaptive exploration and population shift. Strict
zero recall margin is intentional: no tolerated recall loss was supplied.

Additional research safeguard: mean repeat recall in active/inactive groups and
each snapshot must not decline by more than 5 percentage points. Report all
slice counts including small ones. This margin is a research default, not a
business requirement, and does not imply subgroup noninferiority. Report annual
order bins (1,2,3+), span bins (0–90,91–180,181+), and each discount-missingness
slice descriptively. No subgroup is removed from evaluation.

Report within-fold AP/AUC and tie-preserving descriptive FP burden at 90/95/100%
recall overall and active, plus mixed-model pooled curves separately. These
outcome-derived points are descriptive, never candidate cutoff choices.
Report changed alerts/churn catches and errors gained/lost per repeat; saved
inner/outer cutoffs and seed score ranges expose transfer/refit sensitivity.
Compare current baseline predictions with the historical saved baseline and
report discrepancies. No original 0.06 policy enters acceptance.

Every planned arm/fold/repeat and failure is retained in a new output directory.
If any acceptance requirement fails, reject the proposed improvement; do not
retune it to these results. Review evidence before choosing further work. October
outcomes remain excluded. Independent evaluation needs an untouched cohort and
explicit approval after a complete method freeze.

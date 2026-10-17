# Fixed crossfit percentile alignment — frozen before fitting

2026-10-17. No accepted replacement exists. The fixed training-size probe fails
its declared all-repeat AP/oracle monotonicity check; do not declare insufficient
data the cause or pick a reduced-size winner. Post-hoc descriptive observation:
churn and nonchurn score medians increase with training size in all3 repeat
summaries. Membership/vocabulary also change, so this is not causal evidence.
Exact-catch oracle ranking generally improves while deployable inner-selected
rules fail. A common monotone transform cannot change threshold decisions;
aive sigmoid/isotonic calibration of identical scores is not a remedy.

Hypothesis: aligning each component score to an independent TRAINING-reference
population before ensemble averaging may improve transfer of the pooled inner
cutoff. Test one symmetric two-half procedure plus its exact raw-component
control, not a normalization/sampling grid. Use fixed cadence12 inputs (ten+
overdue+gapCV),7 leaves,L2=1, original other HGB parameters, balanced fitting
weights/native missing values. This simpler verified feature set avoids adding
sparse SKU vocab changes to the alignment question. Current product/price/status
fields remain excluded.

Within EVERY inner-training and outer-training partition, construct2 stratified
reseller-grouped halves, shuffle seed42. Fit each half, use the complementary half
as an out-of-fit reference. Carry all reseller snapshots; no fit/reference/group
leakage. Reference labels do NOT fit the mapping (labels only enter the fixed
training stratified split). For each original model seed, predict reference raw
probabilities. At each unique reference score assign empirical midpoint rank
(cumulative count - half tie count)/reference size. Interpolate linearly between
those knots; below/above the reference range map0/1. Raw ties remain tied; new
endpoint ties are retained. Map held-out prediction scores using those knots.
Scores are relative percentiles, NOT calibrated churn probabilities.

Two fully frozen candidate procedures, sharing the exact600 component fits:
1. Mean of the10 raw half-model probabilities (2 halves ×5 original seeds).
2. Mean of their10 reference-percentile scores.
Raw control distinguishes alignment from changed fitting samples/ensembling;
report both even if adverse. No half-count/fraction/reference-rule/seed search.
Symmetric halves use every training reseller for fitting once and reference once;
this is not retuning the earlier bootstrap fraction or deploying oracle cutoffs.
Save component fit/reference/validation identities, reference and raw/mapped
validation scores, training hashes and estimator hashes before assessment.

Use original301/919/1201 outer partitions and saved3-fold inner membership.
Original balanced HGB10 remains PRIMARY comparator with unchanged cached scores
and cutoffs; cadence is secondary only. Inner OOF cutoffs for ALL procedures
maximize precision with recall strictly>90%, ties recall then higher cutoff.
No outer-label threshold/model selection or natural evaluation rebalancing.

Acceptance for EACH candidate unchanged: recall>=original baseline AND FP<baseline
EACH repeat, >=5% mean FP reduction, <=5pp activity/snapshot mean recall loss.
Require paired5000 reseller-cluster-draw lower recall>=0 and FP-reduction>0,
seed2026, tail.05/(27*2) including both procedures. If ALL primary checks pass,
confirm same signs at50000 draws same seed/tail. Report all folds/subgroups,
ranking near required recall, raw-versus-percentile differences and uncertainty.
All margins are research defaults; extreme-tail bootstrap and family accounting
do not erase adaptive research, retraining/reselection or population shift.
No October outcomes, independent evaluation, model promotion or deployment.

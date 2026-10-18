# Period/class-balanced fitting loss — frozen 2026-10-18, cycle 6

Observed development weakness:calibrated and complementary mean procedures cut
FP but repeatedly exchange period losses;pool precision-maximizing thresholds do
not ensure period safeguards. Inverse-reseller weighting failed,but addresses a
different training target. Hypothesis:equal fitting influence for each prescribed
period/outcome cell may reduce domination by common period/class patterns while
retaining one period-free scoring function. No claim this solves forecasting,
causal shift or all worst-group optimization.

ONE candidate:original10HGB,original31leaves/L2=0,all other original parameters,
five original seeds,native missingness. Change ONLY sample weights. For each
FITTING partition,require all4 approved snapshots and both outcomes at each.
Let n_{s,y} be fitting rows in that snapshot/class. Set w_j=N/(8*n_{s_j,y_j}).
Each of8cells sumsN/8,each classN/2,each periodN/4,totalN. No iterativeweight
learning,period/cell selection,cap/exponent/parameter grid or separate models.
Snapshot and labels enter FITTING weights only;StoreID/snapshot are NOT predictors.
Weights use no innerheld/outer label counts. Keep every natural validation row.
This changes the fitting target,not the evaluation distribution or labels.

Use verified cadence-stability original baseline/cached fits and exact301/919/1201
x5outer/3innerseed78 populations;1375rows/212churners/422resellers. Group isolation
at both levels. Same cutoff objective in both arms:pooledinnerOOF precision max
with recall>.90,tiesrecallthenhighercutoff,>=alert,tiesintact. Original fixed.06
and historical precision-floor methods remain separate.300newHGBfits,300cached.

All acceptance rules unchanged:baseline recall preserved and fewerFP EACH repeat;
>=5%meanFP reduction;period/activity/intersection <=.05recallloss,ZEROaddedFP EACH
repeat;undefined/missing/infeasible cases block. Paired5000reseller-cluster draws
seed2026 carry all snapshots/repeats;cycle6spend.00003125 across2metrics/both tails
->tail.0000078125. Require recalllower>=0,FPreductionlower>0;50000same-seed/tail
confirmation ONLYifALLprimarygates pass. Extreme-tail estimates are imprecise,
and no conditional interval covers adaptive research,retraining or calendar shift.
No post-result weakening,moredraw rescue,weight/threshold/period tuning.

Save exact per-fit weights/cell totals/model parameters/training hashes,seeds,
inner/outer scores/cutoffs,before-assessment code/input/runtime snapshots,full
metrics/undefined flags,ranks/tied descriptive burden,transfer tables. Independent
verifier reconstructs all300cell weights/class totals,partitions,30cutoffs,
confusion counts,bootstrap and full acceptance without fitting. Synthetic checks
cover unequal cells,normalization,training-only counts and missing-cell rejection.
No test/source expansion,full-data training,promotion,push or merge.

Primary context:Sagawa,Koh,Hashimoto,Liang(2020),Distributionally Robust Neural
Networks for Group Shifts,https://arxiv.org/abs/1911.08731 and authors'implementation
https://github.com/kohpangwei/group_DRO . They distinguish reweighting and groupDRO
and demonstrate regularization dependence in neural tasks. This fixed equal-cell
HGB loss is a simpler reweighting comparison,NOT their dynamic worst-group method;
their gains/assumptions are not evidence that this churn candidate will work.

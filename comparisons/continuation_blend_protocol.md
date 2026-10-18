# Fixed complementary procedure ensemble — frozen 2026-10-18, cycle 5

Development-only motivation:all four continuation candidates FAIL complete
acceptance. Calibration reduces meanFP14.80% but loses recall/period gates;
activity experts reduce4.05%,fail repeat recall/FP/practical/uncertainty. Experts'
April descriptive tied ranking burden improves all repeats,while calibration's
larger gains are outside April. A no-fit overlap audit finds calibration-only
churn catches7/7/16 and expert-only12/4/9 by301/919/1201,mean within-fold midrank
rho.85957. This supports a bounded complementarity test,not success or a guarantee
that averaging catches the union. Earlier50/50 baseline/cadence trial used entirely
different components and failed;do not retune its weights or rewrite its decision.

ONE primary procedure:fixed arithmetic50/50 mean of retained_logistic calibrated
scores and analytically weight-corrected activity_experts scores. No model,
calibrator or source fitting;reuse ALL saved components,models/maps,keys,folds and
baseline identities. For every expert scoreq,letpi be NATURAL churn prevalence
in the ENTIRE corresponding FITTING partition used for its global class weights.
Set corrected(q,pi)=q*pi/[q*pi+(1-q)*(1-pi)]. Innerpi excludes that innerheld reseller
group;outerpi uses outertraining only. It does not use held labels,group prevalence,
or future data. Original expert weights were global,so use globalpi for both
routes. Preserve0/1endpoints. No unknown-target-prior estimation or EM.

This undoes the odds multiplier of balanced loss IF scores estimated the weighted
posterior;HGB may be miscalibrated,so treat it as a fixed analytic normalization,
NOT verified natural probabilities. Calibration component's disjoint-reference
labels were unweighted. Combining differently scaled raw scores without accounting
for that difference would confound a nominal50/50 mean. No blend-weight,logit-mean,
calibrator,reference,sampling,feature or threshold grid. Report corrected-expert
scores with their SAME training-only cutoff objective as a MECHANISM CONTROL;
it is not eligible for candidate selection or another chance to accept a method.
Rawexpert and calibration component results remain preserved references.

Primary baseline original10balancedHGB unchanged,same1375/212/422 cases,four
approved snapshots,301/919/1201 x5outer,3innerseed78. Both arms independently
precision-max innerOOF recall>.90,tie recallthenhighercutoff,>=alert,tiesintact.
All reseller snapshots isolated,no held label selects components,prior or cutoffs.
No evaluation rebalancing,model loading,protected cohort access or data extension.

Acceptance UNCHANGED:recall>=baseline,FP<baseline EACH repeat;>=5%meanFP reduction;
each repeat's snapshot/activity/intersection <=.05recallloss,ZEROaddedFP;undefined
recall/infeasible/missing fold blocks. Paired5000reseller-cluster drawsseed2026
carry all snapshots/repeats/arms. Cycle5 prospective alpha spend.0000625,two
endpoints/both tails -> .000015625. Mechanism control has no acceptance opportunity.
Require recall lower>=0,FP-reduction lower>0;50000same-seed/tail confirmation ONLY
ifALLprimarychecks pass. No increase of draws can rescue failure. Very extreme
5000draw tails are imprecise;adaptive reuse,retraining,reselection,calendar shift
and source-recording assumptions remain outside intervals. Margins are research
choices,not user business requirements. No lowering after results.

Save pre-execution source/code/protocol identities,each fittingpi and case mapping,
component/corrected/blended scores,control results,innercutoffs,component sizes,
all folds/repeats/periods/activity/intersections,ranking/tied descriptive curves,
transfer tables and independent reconstruction/acceptance checks. Zero newHGB
fits or calibrator fits. No promotion,push,merge or final evaluation.

Primary method context:Saerens,Latinne,Decaestecker(2002),Adjusting the Outputs of
a Classifier to New a Priori Probabilities,
https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf .
Only elementary known-weight odds adjustment is used here,not their EM estimator;
its probability assumptions need not hold for HGB. General retained-calibrator
ensemble mechanics:https://scikit-learn.org/stable/modules/calibration.html .

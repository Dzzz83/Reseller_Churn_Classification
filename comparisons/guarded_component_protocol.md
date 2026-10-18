# Training-guarded component procedure — cycle 9, frozen 2026-10-18

## Observed weakness and hypothesis

The fixed retained-logistic component reduces development FP by 14.80% on average
but loses baseline recall in repeat301 and increases April FP. Cycle8's historical
feature failed too. The inner selection objective so far only required recall>.90;
it did not require estimated recall parity with the baseline or period/group FP
safeguards before transferring a cutoff. Hypothesis: a training-only gate can
avoid using a component where that common goal is not supported by inner evidence.
This is a new complete selection procedure, not reinterpretation of any failed
fixed model, cutoff adjustment from outer labels, or a guarantee about transfer.

ONE component alternative: the already frozen retained_logistic procedure from
cycle1. PRIMARY baseline remains original balanced HGB10. No model pool expansion,
blend weights, new features, calibration parameters, splitter change or fitting.
Use exact saved component inner/outer scores, partitions and model/map identities.
The calibrated component uses cadence12; its fitted pairs/maps remain fixed.
The procedure's scoring schema is their fixed union, with no ID/date predictor.

## Common training-only cutoff objective and explicit fallback

For EACH outer-training partition:
1. On its baseline innerOOF scores select the existing precision-max cutoff with
   recall strictly>.90, ties recall then higher cutoff. This is the reference.
2. Apply ONE common constrained cutoff objective to BOTH original baseline and
   retained-logistic scores: maximize precision, ties recall then higher cutoff,
   subject to pooled innerTP >= reference baselineTP and recall>.90; EACH snapshot,
   activity and snapshot/activity cell must have FP <= reference baselineFP and
   lose at most .05 recall. Use natural matched rows and preserve score ties (>=).
   Undefined subgroup recall makes the alternative ineligible, not silently safe.
   Save every cutoff's feasibility and all selected subgroup counts.
3. The baseline's reference cutoff is feasible under its own constraints and must
   reproduce exactly; otherwise stop for an implementation error. No new baseline
   policy or weakened comparator is introduced.
4. Use retained-logistic only if a constrained cutoff is feasible AND its pooled
   inner FP reduction is at least5% and strictly positive. Otherwise use the
   ORIGINAL baseline model AND original cutoff for the ENTIRE outer validation fold.
   This fallback is a frozen part of the new procedure, never dropping a difficult
   account/fold or pooling only favorable results. Report why each fallback occurred.

Subgroup recall comparisons use integer counts with allowance .05 * churners,
avoiding floating subtraction at the exact five-point boundary. No outer case,
label or prediction influences eligibility, component choice, or cutoff. Freeze
selection before opening saved outer predictions in each fold. No per-customer
routing or data-dependent group definitions. Every outer case is scored naturally.

## Evaluation and complete acceptance

Exactly the same1375approved rows/212churn cases/422resellers,301/919/1201 x5outer,
three inner grouped folds seed78. Every reseller's snapshots remain together at
all model/map/cutoff boundaries. Both procedures share the cutoff objective above.
Grouped evidence is mixed-period unseen resellers,not historical forecasting.
Repeated use of previously inspected development evidence remains exploratory.

All OUTER acceptance unchanged: recall>=baseline ANDFP<baseline EACH repeat;
>=5%meanFP reduction;each repeat's snapshot/activity/intersection <=.05recall loss,
ZEROaddedFP;undefined/missing cases block. A fallback-only result equals baseline
and FAILS the improvement objective. Report all15 choices and every repeat/fold/
period/activity/intersection,TP/FP/FN/TN,recall,precision,alerts,undefined/sparse flags.
Ranking/tied outcome-derived points remain descriptive. Preserve all infeasible
settings and every rejected alternative,not just selected scores.

Primary5000paired reseller-cluster draws seed2026,carrying snapshots/repeats/arms.
Continue alpha spend cycle9=.00000390625,tail=.0000009765625. Require recall lower>=0
and FPreduction lower>0. No failed primary can receive extra-draw rescue. If ALL
primary gates pass,require same-seed/tail50000confirmation AND the additional
fixed ceil(20/tail)=20,480,000-draw simulation-resolution audit from cycle8,with
one-sided exact binomial upper bounds at MC error tail <= bootstrap tail for
negative recall/nonpositiveFP-reduction draw probabilities. Simulation precision
is not new customer evidence or a guarantee of adaptive/future coverage. Margins
are research defaults,not user-supplied business requirements.

## Correctness, identities and reporting

Before construction,test intact ties,common-baseline parity,exact recall boundary,
missing subgroup handling,infeasible/practical fallback and inner-label-only
selection. Independent verifier uses exhaustive tied-cutoff enumeration (separate
from vectorized selector),reconstructs every choice/cutoff/subgroup gate,checks
source/model/map/input identities,baseline parity,all counts and complete outer
acceptance without fitting. No estimator loading. Save source/code/runtime hashes,
all scans,choices,cutoff transfer and component provenance before assessment.
No model promotion,full-data final prediction,protected evaluation,push or merge.

Method context:training-only threshold tuning and nested validation are supported
by official sklearn guidance:
https://scikit-learn.org/stable/modules/classification_threshold.html . This
custom baseline-relative guard is a research rule;the documentation does not
establish its churn performance or prove that an inner constraint transfers.

Implementation clarification before execution: if a reference training subgroup
has undefined recall, the guard is not evaluable and the procedure falls back to
the original baseline for that fold with an explicit status. Baseline self-parity
is asserted on evaluable references. This does not treat an undefined comparison
as safe or waive any outer acceptance gate.

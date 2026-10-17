# Fixed bagging comparison — frozen before fitting

2026-10-16; autonomous backup-only research continuation. The fixed HGB/cadence
blend improves operational point results and its conditional uncertainty bounds,
but its 4.25% mean FP reduction misses the predeclared 5% practical requirement.
Its failed decision and fixed 50/50 weight remain unchanged.

## Hypothesis and previous evidence

Both HGB components have strongly correlated ranks (mean within-fold rho 0.879).
The source of remaining variance may be sensitive boosted partitions; a fixed
bagged learner is a specific alternative inductive bias worth comparing on the
same ten inputs. Earlier seven/ten-feature RF comparisons used the old 30%-
precision objective, and the separate revenue study used five inner folds under
that objective. They did not establish RF's value under this matched high-recall
procedure. No October errors select this trial. The old seven-feature RF versus
ten-feature HGB results cannot isolate architecture; this new comparison holds
the ten input columns fixed.

Freeze two comparisons against the unchanged original ten-feature HGB:

1. Random Forest on the identical ten features: 300 trees, max_depth=None,
   min_samples_leaf=1, max_features='sqrt', bootstrap=True, class_weight=None,
   mean probabilities across seeds 42,78,88,1034,2026. Use the existing ModelFactory
   median imputer with missing indicators, fitted within each training fold;
   balanced sample weights derive only from training labels. n_jobs=1 is an
   execution/resource choice. This compares complete learner/preprocessing
   procedures, not an isolated mathematical tree effect.
2. One fixed equal-weight mean of original HGB and that forest's five-seed mean
   scores. No other weights, transforms, calibration or hyperparameters.

The blend tests whether architecture diversity contributes beyond the two
similar HGB methods. Neither comparison is selected after seeing RF outcomes.
Report both, including failures. Do not later add RF feature sets or weights
within this protocol.

## Shared validation and cutoff objective

Use all 1,375 cases, 212 churners, 422 resellers and the exact saved outer/inner
memberships in results/cadence-stability-20261016 (seeds 301,919,1201; five outer,
three inner; seed 78). Reuse its verified original HGB scores and model/training
identities, because relevant baseline data, code, dependencies and conditions
are unchanged. Fit only the new forest: 15*(3+1)*5=300 estimator fits, not another
baseline rerun. The forest's imputer/weights/model see training rows only, and
all snapshots of a reseller remain together. Validation populations are natural.

Each candidate cutoff maximizes inner OOF precision with recall strictly >90%;
ties favor recall then higher cutoff. Reproduce the baseline cutoff. Blend only
matching component inner scores, never outer labels. Both candidates use exactly
the same rows, labels, splits and objective. The original 0.06 rule is excluded.

## Acceptance, uncertainty and reporting

For **each** fixed candidate, require recall>=baseline and FP<baseline in every
repeat, plus >=5% mean FP reduction. Require no activity-group or snapshot mean
recall loss greater than 5 points. These practical/subgroup margins remain research
defaults, not business requirements; no criterion will be changed after results.
Report all folds, subgroups, annual-count/span/discount-missingness slices,
changed decisions, cutoff/seed sensitivity and high-recall ranking with ties.
AP, calibration or outcome-derived curve points cannot establish acceptance.

Use 5,000 paired reseller-cluster draws, seed 2026, keeping each reseller's snapshots
and repetitions together. Recall-difference lower bound must be>=0 and FP-
reduction lower bound>0 at tail 0.05/(19*2), counting the fourteen old ablations,
two breadth trials, prior fixed blend and these two procedures for two endpoints.
This conservative family threshold does not erase older/adaptive project-wide
exploration, retraining/reselection or population-shift uncertainty. These are
exploratory development findings, not independent confirmation.

Save component/combined inner and outer scores, fitted identities, imputer
statistics, training hashes, all cutoffs, frozen protocol and hashes. Verify
score construction, inner selection, isolation, imputation and metric accounting
without refitting. Preserve all old evidence and the operational pipeline.
No October outcome access, new independent evaluation, promotion or deployment.

# Predeclared cadence stability follow-up — 2026-10-16

Motivation: a post-hoc screen of all fourteen existing ablation configurations
finds f29 meets baseline recall and fewer FP in every original repeat, with
5.19% mean FP reduction. It retains gap CV and overdue ratio, seven leaves and
L2=1, omitting annual order count. A 5,000-draw reseller-cluster interval with
conservative adjustment across fourteen configurations and two endpoints has
recall difference -0.18 to +8.37 percentage points and FP reduction +0.98 to
+56.02. The lower recall bound crosses zero. This is promising old evidence,
not a newly accepted predeclared winner or independent confirmation.

Freeze one fixed configuration against the unchanged ten-feature baseline.
Use **new outer split seeds 301, 919 and 1201**, each five reseller-grouped folds,
and the same three inner folds with seed 78. These are research defaults,
chosen without generating or inspecting their outcomes. The population remains
the same 1,375 development cases, so new partitions are a stability check on
reused data, not fresh independent data. No new feature or parameter grid.

Baseline: original ten inputs and original parameters. Candidate: the ten plus
`overdue_ratio` and `gap_cv`; max_leaf_nodes=7, l2_regularization=1. All other
parameters, training-only balanced weights, native missingness and the five
model seeds are unchanged. Inner OOF precision is maximized subject to recall
strictly >90%, ties recall then cutoff. All reseller snapshots stay together.
No outer-label cutoff adjustment, full-data fitting or October access.

Operational acceptance remains candidate recall >= baseline in every repeat,
fewer FP in every repeat and >=5% mean FP reduction. The activity/snapshot mean
recall-loss safeguard remains <=5 points. Five percent and five points are
explicit research defaults, not business requirements. All folds, repeats,
behavioral/missingness slices and ranking diagnostics are reported.

Use the same 5,000 paired reseller-cluster draws, seed 2026, keeping snapshots
and repeat predictions together. Require a nonnegative recall lower bound and
positive FP-reduction lower bound at tail 0.05/(14*2) for the previously inspected
family. This conservative threshold does not erase adaptive research, model
retraining/reselection uncertainty or source/population shift. Failure of any
requirement means no accepted improvement under this protocol. Do not add more
seeds, alter the objective or choose only favorable repetitions afterward.

Reuse shared fitting/inner-selection code. Its adaptive-selection output is an
unused diagnostic byproduct of the shared evaluator, not another competing
procedure or winner; all fixed-arm results are retained. No saved old experiment
is rerun merely to read findings. This new run addresses the specifically
unresolved split-stability question.

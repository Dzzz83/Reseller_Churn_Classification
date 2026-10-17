# Fixed baseline/cadence blend — protocol frozen before blended outcomes

2026-10-16. The user directed continued autonomous work with CompanyX.bak as the
only source. This trial uses established inputs only; unresolved archive fields
and current profile/status attributes remain excluded.

## Motivation and fixed procedure

Saved complementarity diagnosis on seeds 301/919/1201 finds baseline-only catches
4/0/3 churn cases and cadence-only catches 10/12/13. Baseline-only FP are 84/82/85,
cadence-only FP 56/58/69. Within-fold overall score rank correlation averages
0.879. The models overlap strongly but are not identical. This motivates one
bounded fixed blend; it does not prove that blending will help.

Baseline: original ten-feature boosting. Component: original ten plus overdue
ratio and gap CV, seven leaves, L2=1, other parameters unchanged. Candidate score
is exactly **0.5*baseline probability + 0.5*cadence probability**, with equal
weight on the two existing five-seed ensemble scores. No other weights, ranking
transforms, calibration, hyperparameters or combinations are searched. Arithmetic
averaging is a fixed scoring rule, not a probability-calibration claim.

Reuse the checked fits and scores in results/cadence-stability-20261016. Within
each saved inner fold, both components were fitted on the same reseller-isolated
training rows; require their inner assignments/identities to agree. Average their
OOF scores, then select a candidate cutoff using only those inner labels with
the **same** objective as baseline: maximize precision subject to recall >90%,
ties favor recall then higher cutoff. Reproduce the saved baseline cutoff as a
guard. Average matching outer component scores; outer labels never choose weights
or cutoffs. No new estimator fits are necessary. Never use the original 0.06 rule
as this comparator or alter natural evaluation populations.

## Assessment and acceptance

Use all 1,375 cases, 212 churners, 422 resellers on identical saved three repetitions
of five folds (301,919,1201). Report both primary arms and retain the unchanged
cadence result as a separately labelled component reference. These outer outcomes
have been inspected; this is explicitly exploratory procedure development and
cannot be presented as new independent validation.

Require candidate recall>=baseline and FP<baseline in **every** repetition,
plus >=5% mean FP reduction. Allow no more than 5 percentage points mean recall
loss in either activity group or any snapshot. The practical/subgroup margins
remain research defaults, fixed before blended results. Report all folds and
annual-count/span/discount-missingness slices, changed cases, cutoff/seed ranges,
within-fold ranking and tie-preserving descriptive high-recall burden. AP or
outcome-derived operating points do not establish acceptance or set cutoffs.

Use 5,000 paired reseller-cluster draws, seed 2026, carrying snapshots/repetitions
together. Require recall-difference lower bound>=0 and FP-reduction lower bound>0
at tail **0.05/(17*2)**, conservatively counting fourteen inspected ablation
configurations,two breadth trials and this one blend for two endpoints. This is
not a project-wide adaptive-search correction: older experiments, source
exploration, retraining/reselection and population shift remain outside these
conditional intervals. No final model or operational rule is changed if any
criterion fails. Do not change weights, add seeds or lower margins afterward.

All source/implementation/output hashes and each inner/outer score construction
must be saved and independently checked without fitting. A deployment or new
independent final evaluation requires a separate complete freeze and explicit
user authorization; inspected October outcomes remain excluded.

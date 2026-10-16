# Component ablation of the grouped-validation candidate

The user requested a controlled study of what contributed to the precision
improvement. This follow-up uses development validation only. It leaves main,
the operational model, and the previously frozen research candidate unchanged.

## Frozen design

The five binary changes are:

1. Add overdue ratio.
2. Add annual order count.
3. Add purchase-gap coefficient of variation.
4. Reduce maximum tree leaves from 31 to 7.
5. Set L2 regularization to 1 instead of zero.

Fourteen configurations cover the baseline, each individual addition, the
complete cadence/regularization candidate, each component removed from that
complete candidate, cadence alone, and the two regularization changes together.
This is a targeted ablation, not a full 32-cell factorial study or a Shapley
decomposition. It cannot identify every higher-order interaction or uniquely
allocate the previous adaptive procedure's gain among components.

All configurations use the approved 1,375 development rows, 422 resellers, and
212 churn cases. Repeat seeds 42, 78, and 88 each use five outer grouped folds;
three inner grouped folds with seed 78 select each configuration's cutoff.
Every snapshot of a reseller stays in the same fold. Cutoffs maximize inner
pooled precision subject to recall strictly above 90%, with ties favoring
recall then higher cutoff. No outer-validation outcome selects a cutoff.

Training-only balanced weights, original five model seeds, other boosting
parameters, feature formulas, and native missing handling remain fixed.
Each configuration's observed decision effect includes its own training-only
cutoff adjustment. Average precision (AP) separately describes score ranking.
There is no attempt to choose cutoffs using outer outcomes to force identical
observed recall.

## Reuse, contrasts, and uncertainty

The prior grouped study already fits four identical cells: baseline, cadence
alone, both regularization changes, and the complete candidate. Reuse requires
matching source artifact hashes, shared code, feature order, parameters,
population, and exact outer split indices. Ten new cells need 3,000 estimator
fits; 1,200 prior fits are represented by the four reused cells. These counts
exclude synthetic execution checks and do not imply independent observations.

Seventeen contrasts are fixed before the new fits:

- Five single additions compared with baseline.
- Five full-candidate versus one-component-removed comparisons. Positive effects
  mean the removed component helps when restored to the complete candidate.
- Four bundle contrasts: cadence without/with regularization and regularization
  without/with cadence.
- Cadence × regularization interaction, and leaves × L2 interaction both without
  and with cadence. These are differences in differences.

The primary endpoint is the mean of repeat-level pooled precision differences.
Fifty thousand paired reseller-cluster bootstrap draws carry each reseller's
snapshots and all repetition predictions together. Bonferroni percentile tails
use `0.05 / (2 * 17)` for approximate family-adjusted precision intervals across
the 17 predeclared contrasts. These intervals are conditional on saved fits and
cutoffs; they exclude retraining/reselection uncertainty, are not exact coverage
guarantees, and do not remove prior exploratory-development reuse.

Recall intervals are marginal 95% exploratory summaries. AP differences are
point estimates, without significance claims. Report each repetition's recall
feasibility, every fold/subgroup, and paired caught/missed and false-alert
changes. Do not call a precision gain a useful simplification if it fails the
user's >90% recall criterion. Do not interpret intervals crossing zero as proof
that a component has no effect.

The original model's score vector also supports a cutoff-only comparison:
original 0.06 versus the inner high-recall rule. This reuses identical scores;
AP must remain unchanged. This distinguishes the operating-point tradeoff from
a change in learned ranking.

## Execution and verification

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe -m pytest -c comparisons/grouped_precision_pytest.ini comparisons/test_grouped_ablation.py -q
.venv/Scripts/python.exe comparisons/grouped_ablation.py --smoke --workers 2 --output results/grouped-ablation-smoke-NEW
.venv/Scripts/python.exe comparisons/grouped_ablation.py --source results/repeated-grouped-precision --workers 6 --output results/grouped-component-ablation-NEW
.venv/Scripts/python.exe comparisons/verify_grouped_ablation.py results/grouped-component-ablation-NEW
```

Use new output directories. The JSON protocol is saved before fitting. The run
records code/runtime/input identities, group assignments, inner predictions,
selected cutoffs, outer predictions, model/training identities, fit-reuse
accounting, contributions, and output hashes under ignored results.

The verifier independently recomputes inner cutoffs and reported confusion
metrics/AP, checks group boundaries and populations, verifies exact reused
predictions/cutoffs, and generates the local `report.md`, component effects,
cutoff effects, and paired changes. Six additional synthetic tests cover the
factor grid, contrast definitions, exact configuration matching, interval
adjustment, split mismatch rejection, and population matching. The corrected
synthetic smoke also covers canonical date serialization when combining
loaded and newly generated predictions; its initial failed output is preserved.

This is explanatory, exploratory development reuse. No final test is read, no
new winner is automatically installed, and no push or merge occurs. A simpler
configuration suggested by inspecting these outer results is a hypothesis,
not a prospectively validated replacement.

## Completed results — 2026-10-16

The completed run is `results/grouped-component-ablation/`. All 3,000 new fits
completed, with 1,200 prior fits reused. An independent audit passed all 756
metric rows, population/group checks, inner cutoff recomputation, exact reused
scores/cutoffs, artifact identities, and fit accounting. All 20 focused
study/diagnosis/ablation tests pass. The corrected synthetic smoke, Ruff checks,
format checks, and diff checks pass. The initial smoke exposed mixed string and
Timestamp date serialization; canonical dates were fixed before the real run.

### Which components contributed?

All differences below are mean repeat-level validation effects under the same
inner cutoff selection rule. Positive inclusion effects compare the complete
candidate with the candidate missing that component, so they describe the
benefit of restoring the component. They are not uniquely additive shares of
the prior adaptive selection procedure's gain.

| Component | Precision gain when added alone | Precision inclusion effect in full context | Full-context adjusted interval | Full-context recall effect |
| --- | ---: | ---: | --- | ---: |
| Purchase-gap variation | +1.74 points | +1.14 points | +0.36 to +1.98 points | +0.94 points |
| Overdue ratio | +1.52 points | +0.88 points | +0.17 to +1.63 points | 0.00 points |
| Annual order count | +0.71 points | +0.26 points | -0.28 to +0.83 points | -0.63 points |
| Seven leaves instead of 31 | -0.44 points | -0.08 points | -1.38 to +1.32 points | +5.19 points |
| L2=1 instead of zero | +0.02 points | +0.77 points | -0.003 to +1.52 points | -0.94 points |

Precision intervals are adjusted across all 17 planned contrasts. Recall
effects are secondary descriptive results. Gap variation and overdue ratio
have positive adjusted precision intervals both alone and in the full context.
Annual count's intervals cross zero. L2's full-context result is borderline
after adjustment, so it is suggestive rather than conclusive; its positive
point effect also accompanies slightly lower recall.

Fewer leaves chiefly support recall feasibility here: the full configuration
has **94.65% recall / 29.31% precision**, versus **89.47% / 29.39%** when tree
leaves return to 31 while the other four changes stay fixed. That ablation
fails >90% recall in one repetition. This is an observed operational effect,
not a multiplicity-adjusted confirmatory recall claim.

Neither cadence × regularization nor the two leaves × L2 precision interaction
intervals exclude zero. There is insufficient adjusted evidence for a specific
synergy; that does not prove the components are independent or additive.

### Ranking and cutoff effects

Purchase-gap variation alone increases mean AP from 0.4843 to 0.5107. Overdue
ratio alone increases it to 0.5012. In the full context, however, overdue adds
only 0.0002 AP while increasing operating-point precision. Its decision benefit
should not be described as a large general ranking improvement. Fewer leaves
add 0.0326 AP in the full context, alongside their recall benefit. AP changes
are descriptive point estimates without significance claims.

On unchanged baseline scores, the original 0.06 cutoff produces mean recall
77.36% and precision 32.96%; inner selection for >90% recall produces 91.51%
and 27.16%. Ranking/AP is identical. Retuning that cutoff alone incurs the
recall/precision tradeoff rather than improving learned score separation.

### Development interpretation and subsequent outcome

Gap variation and overdue ratio have the best-supported precision contributions
within this development study. Removing the seven-leaf setting fails the
observed recall objective. Annual count and L2 remain uncertain; their intervals
do not establish uselessness, equivalence or noninferiority after removal.
These results justify retaining the saved candidate as a research artifact,
not promoting it or treating its features as a proven final set.

The subsequent [October evaluation](frozen_october_evaluation.md) finds higher
recall but lower precision than the original historical rule: 97.83% / 20.09%
versus 93.48% / 28.48%. No ablation configuration was evaluated on October to
choose a replacement. The [failure diagnosis](frozen_failure_diagnosis.md)
does not identify one component as the cause. Future comparisons follow
[the research rules](../docs/research_rules.md), using matched cutoff policies
and development-only selection. No feature removal or new winner is approved
by this documentation.

The simplest individual addition with an adjusted precision gain and >90%
recall in all three repetitions is **gap variation alone**: 92.14% recall and
28.89% precision, versus 91.51% / 27.16% for baseline. This is a useful simpler
follow-up hypothesis; it has not been established as equivalent to the full
candidate (94.65% / 29.31%). No single new deployment cutoff is selected.

These fixed-configuration results differ from the earlier adaptive selection
procedure's 91.51% recall / 29.74% precision. The procedure chose configurations
inside each outer training fold; one cannot allocate its +2.59 precision points
by summing these ablation effects.

The detailed local `report.md`, `component_effects.csv`, `cutoff_effect.csv`,
`contributions.csv`, `paired_changes.csv`, and `verification.json` preserve the
evidence. No final testing, automatic model replacement, push, or merge occurred.

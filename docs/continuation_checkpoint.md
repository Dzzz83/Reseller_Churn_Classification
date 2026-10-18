# Current research continuation checkpoint

Updated 2026-10-18. **Current work is complete; no training or evaluation is
running.** The user stopped the earlier autonomous run, then authorized further
comparisons and requested documentation of the findings. This current status
supersedes historical active-work and next-action text below. No candidate has
met complete frozen development acceptance; no operational model is replaced.

Workspace: `research/development-recall-alerts-20261018`, base `db52abd`, with
uncommitted research code/docs. Preserve existing changes, ignored results and
the separate dirty `experimental_research` checkout. No push, merge or deployment.

Latest development: four fixed additions to period/class HGB10 (breadth,
density, personalized recency, annual order control), all failed full acceptance.
Personalized recency improves recent AP 27.64% → 30.92%, mean recall
63.84% → 67.45%, mean FP 217.67 → 213.67, but repeat 919 adds FP and practical,
subgroup and uncertainty gates fail. Saved development source/identities:
`results/development-ranking-information-20261018/continuation_checkpoint.md`.

Latest authorized October evaluation: personalized recency's development-selected
20%–70% targets are added to the original 13 procedures. At 40%,43TP/68FP
versus period/class42TP/51FP. Source, models, cutoffs, predictions, input/runtime
identities and independent audit:
`results/personalized-recency-october-20261018/continuation_checkpoint.md`.
October is previously inspected; no fresh-independent-evidence claim.
The durable [complete table](evaluation_metrics.md#authorized-all-candidate-precision-floor-evaluation--2026-10-18)
and [method inventory](model_and_results.md#methods-tried-and-current-status--2026-10-18)
record current conclusions. Read saved reports rather than rerunning training.

If further research is requested, first diagnose saved inner-to-outer cutoff
transfer for recency repeat 919. Its cause is unproven. Freeze any new policy
before assessment; do not tune on October or relax old decisions. No further
final evaluation or deployment is part of this documentation task.

The chronological sections below preserve earlier instructions, branches and
work states as historical context; they are not current execution directives.

## Objective and boundaries

Fewer false alerts while matching/exceeding baseline recall on identical grouped
development splits. Merely >90% recall is insufficient. Every new comparison has
a predeclared practical/uncertainty/subgroup rule; do not lower it after results.
Current research defaults: recall >= baseline and FP < baseline in each repeat,
>=5% mean FP reduction, <=5-point activity/snapshot mean recall loss, paired
reseller-cluster lower bounds for recall >=0 and FP reduction >0. Conservative
family thresholds and prior adaptive reuse are disclosed per protocol.

Historical branch pipeline_preparation, starting checkpoint de7573499166979c17f829600d9a98b76a2b178b.
Clean initial checkout on an older research branch was switched to this branch.
All current uncommitted code/docs are this task's work. Preserve them and old
outputs/data. No reset, pull, push, merge, deployment or model promotion.
October is the user's test set. On 2026-10-17 the user explicitly authorized
“use that candidate and evaluate on the test set.” The latest cadence/unit-revenue
candidate was frozen using development data and evaluated on October, with no
outcome-driven tuning. The evaluation is complete and fails the objective; see
the latest section below. Do not ask for a different test set or tune on October.

## Runtime and input provenance

All566 handoff files restored/verified. New .venv uses available Python3.14.8 with
all9 principal packages pinned to handoff versions (historical Python3.12.14).
Existing bi_venv untouched. Linux tracked orders are LF; original manifest expects
CRLF. Separate exact-byte copy restores expected hash without editing tracked
CSV or signed records: results/research-provenance-20261016/orders_clean_historical_bytes.csv.
Its line_endings.json records transformation and hashes.

Original diagnosis reproduced in results/baseline-error-diagnosis-20261016-linux-audit:
108 metric rows,15 grouped splits; all6 tables match historical to1e-12. No fits.
1,375 cases/212 churners/422 resellers on four approved snapshots only.

## Completed cycles and preserved evidence

- results/persistent-error-investigation-20261016: predeclared same-snapshot
  missingness-aware ten-feature matching of353 persistent recent FP,11 persistent
  misses,63 inactive continuing buyers to1/3/5 controls. All10 baseline and6 saved
  cadence/spending descriptors reconcile with1,886 accepted orders/28,356 lines.
  Basket-breadth differences motivated two bounded tests, not causal claims.
- results/basket-breadth-20261016-serial: last-order lines;600 fits. Mean2 extra FP;
  recall falls in2 repeats; complete acceptance fails. Initial process-pool attempt
  failed before fits due sandbox forkserver socket, preserved in
  results/basket-breadth-20261016/STATUS.md.
- results/annual-breadth-20261016: annual distinct products;600 fits. FP reduction
  11.33(2.18%), recall falls in2 repeats; acceptance fails. Both original-split
  baseline reproductions match all4,125 original decisions (max score diff1.11e-16).
- results/cutoff-transfer-diagnosis-20261016: no-fit inner/outer and counterfactual
  score-only/cutoff-only decision variation; nonadditive, not causal attribution.
- results/assignment-history-20261016: observed salesperson switches reverse across
  snapshots; most territory/salesperson outcome cells small; no predictor selected.
- results/prior-ablation-operational-audit-20261016: all14 old configurations
  screened without refits. f29 passes new recall/FP point checks but conservative
  recall interval crosses0. Post-hoc screen, not a new accepted winner.
- results/cadence-stability-20261016: f29 (ten+overdue_ratio+gap_cv, seven leaves,
  L2=1, no annual count), new grouped seeds301/919/1201;600 fits. Recall90.72→95.13%,
  FP reduction22.67(4.38%). Recall/FP point checks pass each repeat, but5% practical
  and FP uncertainty fail. Complete report and independent verification passed.
- results/archive-value-investigation-20261016: no-fit numeric discrepancy screen
  using the same fixed matches. Weak/sensitive/reversing differences; semantic/
  version ambiguity remains; no predictor chosen.
- results/complementarity-20261016: no-fit baseline/cadence overlap. Baseline-only
  churn catches4/0/3, cadence-only10/12/13; mean within-fold rank rho0.879.
- results/fixed-blend-20261016: one predeclared50/50 HGB/cadence mean, train-only
  inner cutoffs,0 new fits. FP reduction22(4.25%). All checks except5% practical
  pass; reject complete acceptance, don't retune weights. Independent construction/
  inner-cutoff audit passed. Report present.

All model studies retain per-fold/repeat metrics, subgroups, ranking, changed
cases, identities, protocols/manifests and reports. No operational model changed.
Original implementation snapshots are preserved in results/research-code-*.
The original fixed-blend implementation was recovered to its exact recorded hash
under results/research-code-20261016-fixed-blend before generalizing component IDs.

## CompanyX is reachable (do not claim unavailable access)

Backup /home/dzzz83/Documents/ChatGPT/ResellerChurnClassification/CompanyX.bak,
SHA2561a9f1457f96cf2dc00a4ea4bd8212b74f54337a727cd742722bfec4dd0591379.
Existing companyx-audit container was stopped; we started it. Volume
companyx-audit-data-v2, backup mounted read-only, no published ports.
`debug.companyx_readonly.query` uses sqlcmd inside it; credentials never printed.
Docker needs tool sandbox escalation, already authorized by user's request to
find/query the configured database. Only SELECT queries were executed.

results/companyx-availability-audit-20261016 contains full catalog, SQL, exports,
reconciliation and report. All72 tables lack temporal/CDC/change tracking; no
verified reseller service/payment event log or cancellation transition history.
All1,886 accepted pre-April headers: revision8, ModifiedDate=OrderDate+7days,
20 current cancellations; don't relabel/filter on that current status.
538/1,375 latest headers have current ModifiedDate at/after snapshot, which flags
missing version history, not proven unknown original inputs. Archive accepted
quantities/dates reconcile on unique order/product; line ordinal is not current
detail primary key. ActualCost is not current UnitPrice (median ratio1.2), and
catalog wording conflicts with current trigger semantics; don't call it profit
or a pricing revision. Current profile attributes/price-history revisions excluded.
Original malformed multiline trigger export is preserved; use corrected
order_trigger_lines.csv. Source owner/docs unavailable: user says backup only.

## Latest completed cycles and current active work

Forest RF10 and fixed HGB/RF blend both failed full acceptance. Linear adds302.67
FP/repeat; unit-weight HGB adds9.33, unit-weight RF82.67. Their independent audits
pass and reports are complete under results/{forest,linear,weighting}-comparison-
20261016/. Original balanced HGB remains unchanged. No failed primary result
received50,000-draw confirmation or post-hoc tuning.

results/product-mix-diagnosis-20261016:132 SKU screen, all fixed1/3/5 pairs,
independent2,750-profile source check. Two latest-order SKU contrasts flag across
all4 snapshots, but specific IDs are NOT outcome-selected for prediction.
results/product-mix-comparison-20261017:300 fits, training-only unsupervised
max20 SKU/support20 vocabulary, annual/latest shares+other added to fixed cadence.
TP/FP201/506,202/506,199/491 vs original190/498,195/551,192/505. Recall preserved,
mean17 fewer FP(3.28%), but repeat301 adds8FP and uncertainty fails. Full acceptance
FAILS. Complete report and independent30-cutoff/300-training-vocab/36-metric audit
pass. No50k confirmation. Secondary cadence results do not replace baseline.

results/extended-history-diagnosis-20261017: older observed history only makes
cadence defined for9 extra cases. Predicted overdue directions reverse; CV small/
missing/neighbor-sensitive. No feature selected. Independent1,375 source cases
and21,777 original paired-summary rows reproduce. Helper's old exact code preserved
under results/research-code-20261017-history/ before optional descriptor argument.
results/exact-history-collisions-20261017: only one opposite-outcome pair shares
the specified order-age/subtotal/product/quantity/price/discount signature. Even
that pair has different historical salesperson IDs, so NOT full-source identical
or irreducible. Unique histories are not proof of learnability. Two tests pass.
results/product-mix-transfer-diagnosis-20261017: no universal recall collapse;
nonadditive score-only disagreement exceeds cutoff-only. Both original diagnostic
tables match to1e-12 in results/transfer-refactor-parity-20261017/. No calibration
or outer-label cutoff selected. Source ShipDate delay constant7 days on all1,886
accepted pre-April orders (results/shipping-timestamp-audit-20261017).

Fixed reseller-bootstrap candidate COMPLETED and FAILED: TP/FP199/507,204/520,
205/512, mean5 fewerFP(0.97%), fails per-repeatFP/practical/both uncertainty.
Independent300 resampling/vocab/training identities,30 cutoffs/36 metrics pass;
report complete in results/product-bagging-comparison-20261017. No50k follow-up,
bag-count/fraction/seed tuning. Original unbagged fitter and verifier exact files
are preserved in results/research-code-20261017-{bagging,productmix,product-verifier}/.

Coverage audit: results/product-vocabulary-coverage-20261017/analysis.py learns
NO new vocabulary: uses each saved outer training vocab. Only21.00% annual and
24.54% latest value is explicit on average; medians4.78/3.37%. Outcome associations
are descriptive; not proof coverage causes errors. No cap/support tuning.

Latent5-PC trial COMPLETED and FAILED: TP/FP197/534,202/544,204/523,
mean15.67 extra FP(3.02%); recall point checks pass but per-repeatFP, practical,
both uncertainty fail. Independent300 projection/vocab/input identities,30 cuts,
36 metric rows pass. Complete report in results/latent-mix-comparison-20261017/.
Projection captures mean26.74% outer /28.08% inner training variance. No50k follow-up
or component-count tuning. Preserved root metadata feature enumeration differs
from actual column order; verifier checks SET and hashes use exact identity order.
All source/fit evidence preserved. No promotion/operational model changes.

results/matched-rank-burden-20261017/ audits exact BASELINE catch count. DirectSKU
beats baseline ORACLE FP (not actual cutoff FP) in12/15 outer folds, three losses;
cadence12 wins/one loss/two ties, bagged10/5, latent11/4. All480 inner/outer points
independently reproduce by exhaustive tied-boundary enumeration. No oracle cutoff
becomes operational. This shows descriptive ranking improvement without accepted
policy improvement. It does NOT diagnose calibration/training-size cause. Two
tie/slack tests pass; full report present.

Training-size probe COMPLETED: results/training-size-probe-20261017. Declared
AP monotonicity fails1201; oracle-burden monotonicity fails301/1201. Cannot diagnose
insufficient data or select a reduced-size winner. Both class raw-score medians
rise with fraction in all3 summaries, a post-hoc pattern with membership/vocab
confounds. Independent150 subset/vocab/input hashes,45 metrics/12,375 predictions
pass. Complete report and consistency flags present. No cutoffs/acceptance selected.

Crossfit study COMPLETED and BOTH FAILED: results/crossfit-percentile-20261017/.
Raw TP/FP193/470,202/521,199/493, mean23.33 fewerFP(4.50%); point recall/FP pass,
practical/uncertainty fail. Percentile189/454,199/490,196/456, mean51.33 fewerFP
(9.91%) with positive FP lower21.50, but one baseline-caught churner lost in301
(189 vs190) and recall uncertainty[-2.83,+5.49]pp crosses0. DO NOT waive that loss.
Independent600 component/reference/hash identities,45 cuts/36 metrics per child
pass; reports complete. No50k confirmation or alignment/reference/fraction tuning.

results/cadence-residual-history-20261017 reuses all17 verified descriptors and
same k1/3/5 matching for cadence residuals:360 persistent recent FP,3 persistent
misses(allApril),63 inactive continuing. Latest paid-unit-revenue median contrast
negative for recent FP at all4 snapshots, nearest means negative all4; January
mean reverses k3/5. Inactive direction reverses twice;3 misses unreliable. Partial
motivation only, all adverse results kept. Complete report present.

## Latest completed result — no accepted replacement

ALL fits/audits are complete; no active execution session remains. Latest fixed
procedure: original10 + overdue_ratio + gap_cv + last_order_unit_revenue, HGB7
leaves/L2=1, original5 seeds, balanced training weights/native missing. Raw last
accepted subtotal / accepted summed quantity; no cost/profit/current attributes.
Protocol comparisons/cadence_unit_revenue_protocol.md was frozen before fits.
Output results/cadence-unit-revenue-20261017/ contains complete report/predictions/
inner cutoffs/identities/subgroups/ranking/bootstrap/manifests.

Exact primary baseline TP/FP by301/919/1201:190/498,195/551,192/505.
Candidate:199/471,207/511,203/488. Mean recall90.72→95.75%; FP518→490,28 fewer(5.41%).
All repeat recall/FP,5% practical, subgroup and recall uncertainty checks PASS.
FP-reduction uncertainty FAILS: interval[-2.5122,+60.2277] at tail.05/56,5000 paired
reseller-cluster draws seed2026. Recall interval[+.46996,+10.3751]pp. Thus COMPLETE
ACCEPTANCE FAILS; do not call it a development improvement or promote it. Secondary
cadence196/470,207/527,202/489 is not uniformly worse than added feature. Original
balanced HGB remains primary, unchanged common inner objective precision-max with
recall>90%; historical0.06/October policies stay separate.

300 new fits+300 cached baseline fits. Existing no-fit verifier
`.venv/bin/python -m comparisons.verify_basket_breadth results/cadence-unit-revenue-20261017`
passed15 partitions,30 cutoffs,36 overall metric rows,8250 predictions and all600
recorded cached/new training hashes. It is generic for fixed numeric HGB arms.
Full report already generated; do not overwrite/rerun completed training merely
to read findings. No50k confirmation: frozen safeguard permits it ONLY after ALL
primary checks pass. Do not rescue failed uncertainty by more draws/seeds, a new
tail/margin or post-hoc threshold. Shared numeric runner's synthetic fold smoke
passes baseline parity, group isolation and20+20 identity counts.

## Evidence boundary and next action

No accepted replacement after the completed bounded source/feature/learner/
weight/ensemble/score-transfer cycles. The latest method is promising on point
estimates, not supported by the full frozen standard. No further source-backed
remedy is currently justified by these diagnoses. That is a research judgment,
NOT a claim every possible algorithm/history representation is exhausted.
Do not start an indiscriminate search to clear a near-failed uncertainty margin.
Repeated folds/seeds/draws on these same422 resellers do not add independent data.

Historical recommendation before the explicit October authorization (superseded
for the requested test evaluation): independent reseller cases with complete verified
pre-snapshot histories and six-month outcomes, or an appropriate untouched cohort
whose data roles and complete comparison method are frozen and explicitly
AUTHORIZED before outcome access. A suitable cohort MAY exist inside CompanyX;
none is assumed authorized/independent here. Do not query fresh cohort outcomes
under generic 'continue' authorization. October is inspected and cannot qualify.
The user's protected-evaluation rule explicitly requires authorization; do not
silently bypass it. No new independent evaluation/full-data promotion/deployment
was done. Do not ask again for unavailable source owners/new source: user says
CompanyX.bak ONLY and continue without questions. Requested contact/service/payment
or order-version events are not verified in that source; ActualCost remains
semantically ambiguous. Document any new information before proposing a remedy.

Preserve all current local code/docs/evidence. No commit/push/merge/reset/pull was
performed. Git HEAD remains de7573499166979c17f829600d9a98b76a2b178b on pipeline_preparation;
all current uncommitted research work is this continuation. Historical bundles,
protocols, manifests, archives and raw sources remain intact. No active process
needs polling; use saved reports/manifests. The audit SQL container is available
as documented above, not a new external data source.

## Shared code and checks

Shared paired acceptance/bootstrap/report tables are now in
comparisons/matched_evaluation.py, used by all new studies. No-fit refactor parity:
results/shared-assessment-parity-20261016 reproduces18 tables and every old decision
value across3 completed studies to1e-12. Historical outputs were not overwritten.
Current tests include history boundaries, missingness, ties, cutoff equality,
reseller dependence, blend metadata alignment, training-only RF imputation and a
synthetic nested RF fold proving cached baseline parity/group isolation.

Previous completed full combined suite: **253 passed in60.40s**, including the new
fixed numeric-feature nested smoke. Command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests debug comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py`.
Latest modules/tests pass Ruff; git diff --check passes. Later edits are only
research documentation. Known task-only matplotlib temporary cache was removed;
all datasets, environments, research scripts/evidence and user files retained.


## Latest action: authorized October test — 2026-10-17

Request completed: evaluate the latest candidate, not the old cadence13 model.
Frozen protocol: comparisons/unit_revenue_october_protocol.md. Runner:
comparisons/evaluate_unit_revenue_october.py; no-fit verifier/recovery:
comparisons/verify_unit_revenue_october.py. Evidence:
results/unit-revenue-october-evaluation-20261017/completed_report/report.md.

Candidate13: original10 + overdue_ratio + gap_cv + last_order_unit_revenue;
HGB seven leaves/L2=1, other original parameters, balanced training weights,
native missingness, five-seed mean. Full1375-row development training;
three reseller-grouped inner folds seed78 select precision-maximizing cutoff
subject recall>90%. No October recall target or cutoff optimization. Final cutoff
0.18634604988648462. Original10 matched-policy cutoff0.0067626906041783315.
Forty fits total (30 inner +10 full), all before label access. Case IDs,
models, development OOF, split membership, cutoffs, source/code hashes and
1476 unlabeled predictions were checkpointed before reading supplied outcomes.
All13 development features and ten October inputs reconcile from pre-snapshot
accepted sources. Stores SHA difference is LF/CRLF only; exact historical CRLF
hash recovered without modifying tracked data.

October492 accounts/46 churners:
- Candidate:45TP,194FP,1FN,252TN; recall97.83%,precision18.83%,239alerts.
- Historical0.06:43TP,108FP,3FN,338TN; recall93.48%,precision28.48%,151alerts.
- Original10 matched cutoff policy:46TP,183FP,0FN,263TN; recall100%,precision20.09%.
Thus candidate fails against BOTH references. Against historical:two additional
churners but86 extra false alerts. Against matched policy:one fewer churner and11
extra false alerts. All added false alerts and the candidate miss are among recent
purchasers. Inactive cohort:35churners/10continuing, all alerted by all methods.
Higher aggregate AP does not establish operational improvement.

Paired5000-reseller bootstrap95% intervals: historical recall difference[0,11.36]pp,
FP reduction[-105,-67]; matched-policy recall difference[-7.32,0]pp, FP reduction
[-29,7]. These condition on fitted methods, omitting training/cutoff/source/research
uncertainty. October outcomes were inspected historically; do not describe this
as newly untouched data. No retuning, promotion or deployment followed.

Execution issues preserved: first preparation stopped before fitting because
stores historical SHA uses CRLF; verified byte-equivalence then continued.
Evaluation saved all metrics/cases/paired intervals but optional pandas Markdown
rendering failed (tabulate absent). Preserved partial outputs; no-fit verification
created completed_report, reproducing every confusion row, cutoff, training hash
and serialized-model prediction. As-run source is archived under verification/.
In-memory joblib estimator hashes change on serialization; immutable bundle SHA,
reloaded parameters and prediction parity pass. Runner now uses plain-text tables;
no modeling/evaluation method changed. Original manifest still identifies as-run
source; verifier checks its archived exact bytes rather than replacing history.

No accepted replacement. Requested test evaluation is complete. Preserve the
fixed candidate and both references; any further work must remain development-only
and cannot use these test outcomes to select the next remedy.

Final verification: full combined suite **258 passed in61.51s** using the
recorded combined pytest command; new runner/verifier/tests pass Ruff and
`git diff --check`. New matched-policy baseline scores reproduce saved historical
baseline scores with maximum absolute difference5.421e-20; only their cutoff
policies differ. No active process remains and no disposable files were retained.

## Latest request: diagnose validation design without test access

2026-10-17: user prohibits touching final test. Latest action is a DEVELOPMENT-only
no-fit audit,not another test diagnosis. Do not reopen test datasets,predictions,
labels,bundles or reports for this work. New entrypoint:
comparisons/development_validation_audit.py; full evidence:
results/development-validation-audit-20261017/report.md. All original local changes
and historical outputs preserved. No active jobs,no new fits/cutoff/model changes.

Key findings: grouped splits isolate reseller IDs but mix periods/unmatured labels
at historical prediction dates; average28 fewerFP hides April+33/+24/+30FP losses;
subgroup criterion lacks FP harm constraint; inner→outer candidate FPR38.86→42.11%
while recall91.08→95.75%; adaptive feature research reused same422resellers; five
seeds identical; unit revenue adds only1.33TP/5.33FP gain over cadence on average.
Development recent-buyer prevalence4.18–17.20%. No new evidence proves a coding
or feature-leak bug caused failure. Candidate already failed full acceptance.

Corrective design is prospective forward/matured-label development evidence plus
explicit reseller target population and training-only inner choices,period/activity
harm reporting and cutoff-transfer checks. Do not tune to known test behavior.
Naively dropping ALL future-snapshot resellers yields January76/76positive training
rows; design must address that survival selection. No actual new validation design
or acceptance criteria adopted in this explanatory audit. Independent counts,
manifest checks,Ruff and whitespace checks pass; no test artifacts accessed.

## Latest correction: temporal validation was already tried

User challenged the proposed explanation. Prior controlled selection comparison
in results/validation_strategy_experiment has grouped17/24/34 versus temporal
9/21/32 catches at equal34/68/102 alerts on SAME April2013 development population.
Grouped leads points,formal winner inconclusive. Temporal is not an untried or
established remedy; do not present grouped time mixing as the proven cause of
latest failure. Both views have limitations. Existing inspected October protocols
use grouped selection; a separate temporal-selected October final result was not
located. Read only saved aggregate reports/protocols and code in this follow-up,
no case-level final-test access/refits/new evaluations. Correction is recorded
in docs/debugging.md and journal. No switch of validation design adopted.

## Current checkpoint: development preparation implemented — 2026-10-17

Latest authorized task was to prepare code/docs, not train a new candidate. Done:
- development_checks.py: approved four dates, keys/labels/predictor guards, native
  missingness, grouped isolation, optional six-month label maturity and event times.
- development_reporting.py: every repeat/fold/period/activity/intersection, paired
  harm flags, worst slices, undefined rates, frozen-cutoff transfer/training sizes.
- review_development_study.py: no-fit, explicit fixed_numeric_feature development
  layout, canonical counts, manifest/split/population alignment, immutable output.
- shared assessment and fixed-feature runner: additional diagnostics and optional
  EXPLICIT prospective period_limits; omission retains legacy acceptance. New
  experiments must freeze and pass their limits before fitting. Invalid settings
  fail before fitting. Global acceptance is not weakened.

Owning preparation protocol: comparisons/development_research_protocol.md.
Read it plus updated docs/research_rules.md and pipeline/README.md before the next
experiment. No new candidate or automatic temporal switch has been selected.
Proposed defaults: no additional FP and at most5pp recall loss EACH repeat's
snapshot/activity group, plus original global recall/FP/practical/uncertainty rules.
Freeze candidate, methods, splits, limits and uncertainty family before new fits.
Period/activity intersections and all folds are also reported as diagnostics.
Native source recording/completeness assumptions are not proved by matrix guards.

Final no-fit evidence: results/development-preparation-review-20261017-verified;
report.md, manifest.json and independent_verification.json reproduce252 old metric
rows, exact old decision and450 transfer rows. Initial review directory without
-verified is preserved with its original execution hashes. Do not overwrite either.
No test data/labels/predictions/model bundle accessed, no actual candidate refits or
threshold selection; only synthetic tests and saved development artifact review.
AGENTS.md untouched. No commit/push/merge/reset/deployment. No active model job.
Final test-suite completion and whitespace/lint status follow below.

Final verification: **278 tests passed in64.98s** with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests debug comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py`.
Focused preparation tests20 passed. Changed modules/tests pass Ruff;
`git diff --check` passes. Final review output manifests and current implementation
hashes pass; no disposable task files or running processes remain.

## ACTIVE CONTINUATION — 2026-10-18 (supersedes earlier next-action text)

Branch research/development-recall-alerts-20261018 from db52abd in primary repo.
Experimental checkout c047741 and its ablation edits untouched. No push/merge,
operational changes,promotion or real final evaluation authorized/performed.
Mixed historical docs unexpectedly displayed protected aggregates; boundary
incident recorded in journal,excluded from hypotheses/selection. Read only narrowly
scoped development evidence from now on; do not reopen historical test sections.

Cycle1 retained logistic reference calibration:FAILS despite14.80%FP reduction,
one repeat recall loss and period/intersection harm. Results/retained-calibration-
20261018,independent600identity/540metric/30cutoff/5000draw audit passes.
Cycle2 smooth additive cadence:FAILS,56moreFP on average. Results/smooth-cadence-
20261018-02,60fits/30cuts/540rows/statistics/prediction reconstruction pass.
First interrupted spline run preserved. Separate acceptance verifier independently
reproduces both full failed decisions; no50000draw rescue allowed.

Current hypothesis/next action: execute frozen reseller-weighting cycle3 after
synthetic checks. comparisons/reseller_weighting_protocol.md specifies original10
HGB with inverse FITTING-partition reseller-history weights renormalized to
balanced class totalsN/2. Same splits/cutoff objective/natural validation. No
reseller-count predictor. Freeze and save all weights and identities before scoring.
Continue alpha ledger:.001,.0005,.00025 across cycles,each /4 for two metric tails.
Periods/activity/intersections each repeat <=5pp recall loss,zero addedFP; overall
recall>=baseline each repeat,strict fewerFP each repeat,>=5%mean reduction,conditional
bootstrap lower recall>=0/FP reduction>0. No independent confirmation claims.

Runtime .venv Python3.14.8,sklearn1.9.1;threads OMP/OPENBLAS=1. Commands and outputs
owned by each protocol. New reusable callback in evaluate_fixed_fold supports
non-HGB fitting while retaining cached memberships,metadata and baseline; original
default covered by existing smoke. No completed training should be rerun to read
results. Durable code/journal remain uncommitted on research branch; generated
results ignored. Further work and final verification remain required.

## LATEST ACTIVE CHECKPOINT — 2026-10-18, seven completed cycles

The objective remains **unmet**. No DEVELOPMENT IMPROVEMENT has passed acceptance.
The research goal stays active; no candidate is promoted or eligible for final
evaluation. All seven failed decisions are preserved.

### Workspace and boundary

Primary checkout: `/home/dzzz83/Documents/code/Reseller_Churn_Classification`.
Branch: `research/development-recall-alerts-20261018`, based on `db52abd`.
Research code and documentation remain uncommitted. No push, merge, reset, pull,
commit, operational model change, or final evaluation occurred. The experimental
checkout at `c047741` and its user-owned ablation changes remain untouched.
No training process remains active after cycle 7.

Mixed historical documentation unexpectedly exposed protected test aggregates
near the start. This boundary incident is disclosed in the journal; that material
was excluded from research hypotheses and selection. Do not reopen those sections.
No test case-level files, prediction files, label files, model bundles, or final
evaluators were accessed for these experiments. Prior test inspection prevents
any claim of fresh independent final evidence.

### Current hypothesis and next action

Investigate whether monotonic shape constraints on recency/cadence have defensible
conditional domain support, using primary sources and approved development evidence.
This is a potential regularization assumption, not an established cause or a frozen
candidate. Saved-score associations are not partial dependence or causal evidence.
No further fit should start until the hypothesis and full protocol are justified.
If unsupported, record that result and find another evidence-backed direction.
Do not retune the failed weights, routes, kernels, blend fractions, or cutoffs to
clear a failed margin. No routine user question is needed.

### Completed comparisons

The common baseline has TP/FP of 190/498, 195/551, and 192/505 for repeats
301, 919, and 1201. Its mean recall is 90.72%, with 518 FP per repeat.
All studies use the same natural 1,375 rows, 212 churn cases, and 422 resellers.

| Cycle | Procedure | Mean recall | Mean FP | Result directory under results/ |
| --- | --- | ---: | ---: | --- |
| 1 | Retained logistic calibration | 91.19% | 441.33 | retained-calibration-20261018/retained_logistic |
| 2 | Smooth additive cadence | 92.61% | 574.00 | smooth-cadence-20261018-02 |
| 3 | Inverse reseller-history weighting | 90.41% | 549.00 | reseller-weighting-20261018 |
| 4 | Fixed activity experts | 90.41% | 497.00 | activity-experts-20261018-02 |
| 5 | Fixed complementary mean | 91.19% | 448.67 | continuation-blend-20261018/complementary_mean |
| 6 | Equal period/class weighting | 90.41% | 538.67 | period-weighting-20261018 |
| 7 | RBF cadence margin | 92.30% | 659.67 | rbf-cadence-20261018 |

Every row FAILED complete acceptance. Calibration and blending have favorable
averages but lose baseline catches in repeat 301 and violate period/intersection
gates. No failed primary received 50,000-draw confirmation or a weaker criterion.
The corrected-expert mechanism control in cycle 5 is ineligible for selection.

Completed real fit slots: 1,200 new HGB classifiers, 60 additive logistic models,
60 SVMs, and 600 reference logistic calibrators. One initial spline fit was lost
to a metadata integration failure; synthetic fits are separate. Previously saved
baseline and crossfit models were reused, not retrained to read their results.
Identical five-seed predictions do not supply independent robustness evidence.

### Frozen rules and uncertainty

Keep reseller snapshots together in outer and inner validation. The common cutoff
rule maximizes pooled inner-OOF precision with recall strictly above 90%; ties
favor recall, then the higher cutoff. Scores equal to the cutoff alert. Original
fixed 0.06 and precision-floor policies remain separate historical references.
Grouped validation estimates unseen reseller identities across mixed periods;
it is not chronological forecasting.

Acceptance requires ALL: baseline recall preserved and strictly fewer FP in each
repeat; at least 5% mean FP reduction; at most 5 percentage points of recall loss
and zero added FP in each repeat's snapshot, activity, and snapshot/activity cell;
paired conditional lower recall difference >= 0 and lower FP reduction > 0.
Undefined recall, missing predictions, or infeasible folds cannot silently pass.
These margins are conservative research defaults, not business-approved limits.

Use 5,000 paired reseller-cluster draws, seed 2026, carrying every snapshot and
repeat together. Continuation alpha spend is .001 / 2**(cycle - 1), divided by four
for two endpoints and both tails. Cycle 8, only if justified and frozen, has spend
.0000078125 and tail .000001953125; do not reset the family. The empirical extreme
tails are imprecise and omit adaptive research, retraining/reselection, and future
population/calendar changes. A 50,000-draw check is allowed only after ALL primary
gates pass, with the same seed and tail; it cannot rescue a failure.

### Evidence, repairs, and reproduction

`results/development-continuation-20261018/` contains the report, summary,
repeat-level counts, intervals, runtime inventory, study identities, code snapshot,
and base-commit mapping. Frozen protocols are the seven matching
`comparisons/*_protocol.md` files. The existing research journal and evaluation
metrics own the explanation; generated results stay ignored by Git.

Independent audits passed for every comparison: training/weight/route/support
identities, score reconstruction where applicable, cutoffs, operating counts, and
all seven complete decisions from 5,000 paired draws. Transfer reviews retain
450 rows per reviewed study and actual component sizes. The final focused suite
passed **63 tests in 23.65 seconds**; lint and `git diff --check` passed. SVC emits
one expected future deprecation warning for explicit probability=False. The full
legacy suite was not run because some tests load broader source files; the selected
checks use synthetic or explicitly approved development data only.

The initial spline metadata failure and expert all-missing-column binning failure
are preserved in their interrupted directories. Expert version 2 omits only
entirely missing FITTING columns, recorded in a pre-score protocol amendment;
other missing values remain native. The calibration child preserves its original
self-referential manifest and uses completion_manifest.json for the final outputs.
The blend child similarly binds its post-assessment intersection decision with a
completion manifest. Exact as-run sources and earlier verifier versions are saved.

Development lineage reconciliation proves that ablation, research, and renamed
operational preparation have identical approved keys, labels, ten values, and
missingness under the explicit name mapping. Preserve the historical
`discounted_line_share_12m` schema; operational `discount_frequency_12m` changes
only its name. Numeric parity does not prove historical source completeness.

Use `.venv/bin/python` with OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1.
Every runner requires a new output directory. Read completed reports rather than
refitting. Verifiers perform zero fits but usually refuse to overwrite existing
audit files: use task-local copies for rechecking, retaining the originals.
`verify_development_acceptance` takes directory, --candidate, and --tail from the
frozen protocol. See the consolidated reproduction instructions. No source/data
expansion or protected evaluation is assumed by this checkpoint.


# Continuation checkpoint — 2026-10-18, stopped at user request

The user asked to finish the current work and report the last hour, without requiring
achievement of the research objective. No further research cycle is authorized by
this checkpoint itself. The objective remains UNMET: no DEVELOPMENT IMPROVEMENT
has passed all frozen gates. No candidate is eligible for final evaluation.

## Workspace and boundaries

Primary checkout /home/dzzz83/Documents/code/Reseller_Churn_Classification.
Research branch research/development-recall-alerts-20261018, base db52abd.
Changes remain uncommitted; no push, merge, deployment or operational change.
Experimental checkout c047741 and user-owned ablation changes remain untouched.
Only approved development cases were used: 1375 rows,212 churn cases,422 resellers,
four snapshots July2012/October2012/January2013/April2013. No final evaluation ran.
Earlier mixed historical documents unexpectedly exposed protected aggregates;
that incident was disclosed and excluded from hypothesis/selection. Do not reopen
mixed old journal/checkpoint sections. No additional protected material was opened
in this wrap-up or current cycle. No protected case/prediction/label/model file was
used. Read this standalone checkpoint instead of historical mixed documents.

## Completed evidence

First seven continuation cycles all failed. Their complete report/source/runtime
bundle remains results/development-continuation-20261018, unchanged.

Cycle8: early_quarter_share, originalHGB10 plus one verified historical order-share
feature,300 new fits. TP/FP repeats301/919/1201:190/504,198/553,195/515.
Baseline190/498,195/551,192/505. MeanFP increases6; complete acceptance FAILED.
Independent5500quartercounts/1375features,split/cutoff/count/acceptance audits pass.

Cycle9: fixed retained-logistic component with inner period/activity guards and
explicit whole-fold baseline fallback. No new fits. All15 alternatives infeasible;
all4125 candidate decisions equal baseline,FP reduction0; acceptance FAILED.
Independent exhaustive32690tied-cutoff boundaries and540operating rows pass.
Training-bound diagnosis: April FP constraints bind all15 lower bounds; earlier
period recall constraints prevent a common feasible cutoff. No guards relaxed.

Cycle10: positive_floor_veto over unchanged baseline alerts. No new fits. Floor is
minimum fixed retained-logistic innerOOF score among baseline-caught churners;
equality retained. Floor saved before outer prediction file read. Common cutoff
objective reproduces original baseline cutoff; every outer candidate alert must
be a subset of baseline. No floor grid, margin or outer-label tuning.
TP/FP:188/451,195/519,191/459. MeanFP476.33 vs518 (8.04% reduction).
Recall mean90.25% vs90.72%; loses2 catches in301 and1 in1201 (dependent repeats).
October/recent-purchase recall loses6.67points in301 and1201, exceeding5point limit.
Complete acceptance FAILED; no50000draw or exact-sign rescue was run.
Independent15floors,8250predictions,540operating rows,component identities,
alert-subset property,42safeguard cells and5000cluster draws all reproduce.
Results:results/positive-floor-veto-20261018. Frozen protocol and as-run source
snapshot are inside. Reproduce with the module commands in this wrap-up report.

## Uncertainty and limitations

Same matched grouped outer301/919/1201 x5folds,inner3fold seed78; every reseller's
snapshots stay together. Cutoff precision maximum with recall>.90, ties recall
then higher cutoff. Ten-feature HGB configuration and five seeds unchanged.
C8/C9/C10 tails .000001953125/.0000009765625/.00000048828125; no reset.
Research defaults remain >=5%meanFP reduction,recall preserved andstrictlyfewerFP
everyrepeat,<=.05recall loss andzeroFPincrease everyperiod/activity/intersection,
pairedreseller lowerrecall>=0 andlowerFPreduction>0; undefined fails.

Exact integer-convolution bootstrap sign audit implemented/tested prospectively.
Prior calibration diagnostic recall-negative probability .3407117774 despite tiny
FP-nonpositive probability1.105e-15. C8recall-negative.1556373,FP-nonpositive.8067529;
C9FP-nonpositive1. These diagnostics changed no historical decisions.
Cycle10 additionally froze exact sign counting only after all5k/50kgates pass;
primary failure correctly prevented execution. Exact empirical counting resolves
simulation error only,not retraining,adaptive choice,calendar/future uncertainty.
OOF training-positive floor has no conformal/NP coverage guarantee.

## Next action if research is resumed

Do not retune veto floors or guards to observed outer misses. No pending fit/query.
The next distinct direction would need development-only source/provenance evidence
and a new frozen protocol. Latest-basket discount composition is only an untested
possibility; no feature/query/protocol is frozen. Database container running was
confirmed, but no DB rows were queried in this phase. Its helper's read-only
name/docstring does not enforce SELECT: use explicit bounded SELECT only if needed.
Strict final exclusion remains. Repeated development adaptation can overfit the
same422resellers andfourperiods. New seeds do not create new independent evidence.


# Latest checkpoint addendum — authorized October evaluation completed

2026-10-18: user requested “evaluate on the final test set.” Positive-floor veto
procedure frozen; results/positive-floor-october-evaluation-20261018/summary.md owns
complete results, sources and commands. Candidate100%recall/21.20%precision/171FP;
matchedbaseline100%/20.09%/183FP;historical.06:93.48%/28.48%/108FP. Reused inspected
cohort,not independenttest. Development acceptance remainsFAILED. No deployment.
No active fit or research continuation. Do not tune from these test outcomes.


# Latest checkpoint — all-candidate precision sweep completed, 2026-10-18

User explicitly authorized evaluating other candidates with development precision
targets, after earlier October evaluation. Scope frozen at13procedures x9targets.
All117choices/checkpoints precede this outcome stage. October already inspected;
descriptive reuse only,no test-selected model/threshold/floor. 92feasible25infeasible.
Independent117choices/128fitidentities/7fullbundles/276metricrows pass. No active fit.
No operational model change,push,merge or deployment. Full developmental acceptance
remains unmet for all prior candidates. Do not restart autonomous research or tune
from these outcomes. User owns next decision.

Authoritative report and exact commands:
results/all-candidate-precision-october-20261018-04/report.md.
Full counts and undefined flags: evaluation/comparison.csv andevaluation/metrics.csv.
As-run source,schemas,protocol,cutoff scans,models,input/runtime IDs and audits saved.
Older cadence13 exactspec refitted under active runtime;Windowsmanifest handled by
rootfilehash checks. Attempts01–03 preserved,cache recovery documented inreport.
All historical source features remain strictlypreprediction;no SQL/new labels.


# Latest diagnostic checkpoint — 2026-10-18

User asked whether missed recent-purchase resellers show unused warning signs.
Development-only descriptive audit complete,not a new model experiment. Authoritative
report:results/hard-recent-warning-audit-20261018-02/report.md. 67cases/66resellers,
47have observedwarning,15noneamongdefinedcomparisons,5insufficienthistory. Warnings
also common among continuingbuyers;no clean separator found. Weak revenue-ratio
and product-diversity associations remain exploratory and overlap failed prior
research. NoOctoberinputs,newfits,cutoffchanges,SQL,newlabels or deployment.
6875ratios,603neighborhoods independentlyverified;3synthetic tests pass. All data,
code,protocol andartifactidentities saved. No pending fit/query or autonomous
research continuation. Any model trial requires a separately frozen development
protocol; previously inspected October is not independent confirmation.


# Research checkpoint — development ranking information, 2026-10-18

Current request:find information that ranks churners above continuing buyers
without many additional false alerts. Four fixed development-only comparisons
completed;no active fit/query. Scope included annual breadth,density,personalized
recency,and annual-order attribution control under unchanged period/class HGB10,
same nested301/919/1201x5groups,innerseed78 and40%precisionpolicy. Historical code
schema retained. No new October access/evaluation or operational modification.

Personalized recency is the strongest provisional signal:recentAP27.64→30.92%,
mean gain3.27points,descriptive95%gain interval[.92,5.59]points,positiveall3repeats.
MeanTP gain7.67,FP reduction4(1.84%);repeat919adds4FP,period/intersectionlimitsfail,
operatinguncertaintygatesfail. Thus NOT an accepted DEVELOPMENT IMPROVEMENT or
finalmethod. Both density and annualcount gainrankingpointestimates,butAPintervals
includezero andoperatinggatesfail. Breadth reversesAPgaininone repeat. Allfourfailed
acceptancedecisions preserved;no50k/exactrescue. Cycles11–14 alpha ledger continued,
notreset. 1200fits saved;no fitting recovery or discardedcases.

Authoritative summary:results/development-ranking-information-20261018/report.md.
Studies:results/period-product-ranking-20261018(twoarms),
results/period-personalized-recency-20261018,
results/period-order-exposure-20261018. Every study has protocol,asrunsource,input/
model/weights/splits/scores/cutoffs/runtime IDs and independent verification.
1200modelreloads/120cutoffs/2160metricrows/full5000operatingdrawsperarm verified;
17rankingdraws independently spotchecked perarm. No new finalperformance claim.

If resumed,next diagnose the saved inner-to-outer score/cutoff transfer for
recency repeat919 before changing a policy. Do not choose features/cutoffs from
October,retroactively relax gates,combine everything blindly or treat prior test
asindependent. Current rootresearch branch/db52abd/uncommitted;experimentalcheckout
and userfiles untouched. No push,merge,commit or deployment.

## Latest authorized evaluation — personalized recency, 2026-10-18

User requested October evaluation and addition to existing20–70target table.
Completed; no active job. New self-contained checkpoint:
results/personalized-recency-october-20261018/continuation_checkpoint.md.
Prior development failures preserved. This authorization/evaluation does not
request further model selection or deployment.

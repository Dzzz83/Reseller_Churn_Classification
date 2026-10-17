# Current research continuation checkpoint

Updated 2026-10-17. **Keep working autonomously without further questions.** The
user explicitly instructed this after confirming CompanyX.bak is the only source.
Source limitations exclude ambiguous fields; they do not block all existing-input
research. No accepted development replacement has been established yet.

## Objective and boundaries

Fewer false alerts while matching/exceeding baseline recall on identical grouped
development splits. Merely >90% recall is insufficient. Every new comparison has
a predeclared practical/uncertainty/subgroup rule; do not lower it after results.
Current research defaults: recall >= baseline and FP < baseline in each repeat,
>=5% mean FP reduction, <=5-point activity/snapshot mean recall loss, paired
reseller-cluster lower bounds for recall >=0 and FP reduction >0. Conservative
family thresholds and prior adaptive reuse are disclosed per protocol.

Branch pipeline_preparation, starting checkpoint de7573499166979c17f829600d9a98b76a2b178b.
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

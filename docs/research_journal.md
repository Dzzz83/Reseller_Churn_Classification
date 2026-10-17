# Research journal

## 2026-10-16 — continuation and cycle 1

Started from clean `research/revenue-shap-results-20261016` (90fa7bb), then switched
to user-requested `pipeline_preparation` at de7573499166979c17f829600d9a98b76a2b178b.
Restored and verified all 566 handoff files. No pull, reset, historical overwrite,
October outcome access, promotion or deployment. Recreated `.venv` using available
Python 3.14.8 and the handoff's nine pinned packages (historical Python 3.12.14 is
not installed). The initial sandboxed package download failed DNS; approved
network installation succeeded. Existing `bi_venv` was preserved.

Provenance mismatch: tracked orders have Linux LF bytes; historical manifest
expects CRLF. A separate LF-to-CRLF copy exactly recovered SHA-256
`2f65b53f068dd14080484ce522d08b78ea9ed00328a24a7c70a48a97cd1166e5`.
Tracked data and signed historical records are unchanged. See
`results/research-provenance-20261016/line_endings.json`.

Reproduced diagnosis into `results/baseline-error-diagnosis-20261016-linux-audit/`:
108 metric rows, 15 outer partitions, 1,375 rows/212 churners/422 resellers;
no fits. Current combined suite passed 210 tests. Historical claims about import
collisions are superseded by the checkpoint repair; older >90% objective is
superseded by baseline recall preservation and fewer FP.

Completed the predeclared matched-history investigation, with four new passing
synthetic tests. Protocol and findings:
[Persistent error investigation](../comparisons/persistent_error_investigation.md).
Outputs: `results/persistent-error-investigation-20261016/`; manifests capture
inputs, implementation hashes, runtime and output checksums. Ten baseline and
six history descriptors reconcile with saved inputs. Directional basket-breadth
signal motivates one minimal model trial; other descriptors overlap, reverse
across snapshots, or repeat already tested hypotheses. This is exploratory,
with dependent/reused controls and no causal or feature-utility claim.

Cycle 1 established no model improvement. Next: the separately frozen
[basket breadth protocol](../comparisons/basket_breadth_protocol.md), then tests,
matched nested evaluation and complete reporting. No new fitted candidate exists
yet. Primary objective and practical/uncertainty/subgroup requirements are frozen
before fitting. No genuine data blocker for this trial.

Before reading cycle 2 outcomes, a second bounded contingency was frozen in
`comparisons/annual_breadth_protocol.md`: if latest-order breadth fails, isolate
annual unique-product count using the same baseline and splits. Both hypotheses
come from cycle 1, and annual variety has not been isolated under this model and
operating policy. No unlimited feature search is planned. The second trial's
uncertainty criterion accounts for two endpoints and the two bounded trials;
prior adaptive exploration still limits inference.

## Cycle 2 — last-order breadth rejected

Completed 600 fits under the frozen protocol in
`results/basket-breadth-20261016-serial/`. Candidate TP/FP by repeat:
42: 194/540; 78: 191/496; 88: 196/532. Baseline: 195/530, 192/497,
195/535. Mean recall change -0.16 percentage points and **two extra FP** per
repeat. Recall is lower in two repetitions. Conditional paired reseller-cluster
95% intervals: recall change -1.91 to +1.52 points; FP reduction -14 to +10.
The subgroup safeguard passes, but every other acceptance check fails. Reject
this addition; matching associations did not demonstrate operational utility.

Read-only independent verification reproduced all 30 inner cutoffs and 36
pooled/fold overall metric rows, checked 15 grouped splits and 600 recorded
training identities. The current baseline matches all historical 4,125 decisions;
maximum score difference 1.11e-16. No final outcomes or protected evaluation.
The code used for this cycle is additionally preserved with hashes under
`results/research-code-20261016-cycle2/` before extending the runner for the
predeclared annual-breadth contingency. The shared fitting code is unchanged.

The expanded combined suite passed 218 tests. After adding annual-feature
selection and its predeclared stricter interval tail to the same runner, all four
focused basket-study tests passed. Cycle 3 now runs in
`results/annual-breadth-20261016/`, using the contingency protocol frozen before
cycle 2 outcomes were read. No last-order retuning or combined-feature search.

## Cycle 3 — annual breadth rejected; no-fit follow-ups

Completed another 600 fits in `results/annual-breadth-20261016/`. Candidate TP/FP:
42: 196/527; 78: 189/484; 88: 194/517, versus baseline 195/530, 192/497,
195/535. Mean FP reduction 11.33 (2.18%) but recall falls in repeats 78/88,
and repeat 78 recall is 89.15%. The predeclared 1.25–98.75% conditional intervals
are recall difference -2.81 to +1.88 points; FP reduction -2 to +25. The practical,
recall and uncertainty requirements fail; subgroup safeguard passes. Reject.
The independent 15-fold/30-cutoff/36-metric audit passes. Baseline again reproduces
all historical decisions. Readable complete reports are saved for both trials.
Cycle-3 source snapshot: `results/research-code-20261016-cycle3/`.

The no-fit transfer diagnosis at `results/cutoff-transfer-diagnosis-20261016/`
finds baseline active cross-repeat decision disagreement 12.39%, cutoff-only
4.29% and score-only 11.58%. These are nonadditive counterfactual diagnostics,
not causal shares. Both breadth candidates have similar variation. Equal-fold
inner/outer averages do not show a universal collapse of recall after inner
selection; they cannot certify full-data cutoff transfer. No new cutoff chosen.

The additional prespecified historical-assignment screen at
`results/assignment-history-20261016/` finds changing direction across snapshots
for observed salesperson changes. Active territory cells are small in 28/32
snapshot/category cells; 54/55 active salesperson cells are small. No new
assignment predictor selected. Source availability remains assumed at order time.

A no-refit screen of **all fourteen** old ablation configurations at
`results/prior-ablation-operational-audit-20261016/` finds f29 meets the current
recall/FP point checks in every original repeat and reduces mean FP by 27 (5.19%).
It retains gap CV, overdue ratio, seven leaves and L2=1, omitting annual count.
A conservative paired bootstrap across the historical fourteen configurations
and two endpoints has recall interval -0.18 to +8.37 points and FP reduction
+0.98 to +56.02. This post-hoc screen cannot be called newly predeclared success.
It motivates a frozen new-partition stability study rather than declaring the
existing data exhausted. Protocol: `comparisons/cadence_stability_protocol.md`.

## Live CompanyX access recovered; cycle 4 stability running

The user corrected the source-availability question: CompanyX access was already
configured. Located the exact backup and existing stopped `companyx-audit`
container, restarted it, and verified SELECT access to CompanyX_Audit. No source
DB was overwritten or restored. The live audit is in
`results/companyx-availability-audit-20261016/`; findings and failed assumptions
are documented in `docs/data_analysis.md`. The absence of CRM/payment events was
checked in the database, rather than inferred from the handoff exports. Archived
transactions expose a cost/current-price discrepancy; its meaning and historical
version availability remain unresolved, so it is not modeled.

The old f29 configuration now has a predeclared stability run on new grouped
seeds 301/919/1201, with paired baseline/candidate inner selection and 600 planned
fits in `results/cadence-stability-20261016/`. These are new partitions of already
inspected data, not independent confirmation. Criteria are frozen and conservative.
Shared fitting utilities are reused. Fifteen focused checks passed before launch,
including analysis with new repeat IDs and a named candidate. No October access.

## User-directed continuation after source limitations

The user confirmed CompanyX.bak is the only source and instructed continued work
without further questions. No source-access blocker exists; archive semantic/
version limitations exclude that field family but do not exhaust established
transaction-model research. Continue autonomously within that scope.

Cadence stability completed: baseline TP/FP301:190/498,919:195/551,1201:192/505;
candidate196/470,207/527,202/489. Mean recall90.72%→95.13%, FP reduction22.67
(4.38%). All point recall/FP checks pass, but practical5% and conservative FP
lower bound(-1.33) fail. Recall interval+0.60 to+8.94 points. All independent
15-fold/30-cutoff/36-metric checks pass. No accepted improvement or promotion.

Archive-value numeric screening in results/archive-value-investigation-20261016
finds weak/neighbor-sensitive or reversing differences. No new source predictor
is selected. Its business/version meaning is still unresolved.

A no-fit complementarity diagnosis in results/complementarity-20261016 shows
baseline-only churn catches4/0/3 and cadence-only10/12/13 on the new repeats,
with distinct false-alert sets. This motivates **one fixed equal-weight blend**,
not an arbitrary weight search. Protocol comparisons/fixed_blend_protocol.md
was frozen before any blended operating results. Both components reuse the
verified existing fits; the blend's cutoff uses training-only combined inner
OOF scores. No new fits or October access. The final combined suite now passes
221 tests. Continue through blend execution, independent verification and reporting.

## Fixed HGB/cadence blend — rejected under the practical requirement

`results/fixed-blend-20261016/`: no new estimator fits. One predeclared50/50 mean
of matching component scores; its own cutoff uses combined inner OOF only.
All30 score constructions/30 cutoffs/15 partitions independently check out.
Every recall/FP point check and conservative uncertainty check passes, but mean
FP reduction22 (4.25%) fails the frozen5% default. Do not retune weights or call
this complete acceptance. No October access or operational model change.

Shared paired assessment was consolidated in comparisons/matched_evaluation.py.
A no-fit replay reproduces18 tables and every saved decision value across all
three previous trials to1e-12 (results/shared-assessment-parity-20261016/).
Twelve focused blend/paired-analysis checks pass. Previous implementation
snapshots remain preserved before refactoring.

The next predeclared trial is comparisons/forest_comparison_protocol.md: same
ten inputs, same grouped folds/objective, fixed baseline300-tree Random Forest
and one fixed HGB/forest50/50 mean. It isolates the input set while comparing
learner/preprocessing procedures, using the established training-only imputer.
Only300 new forest fits are planned; unchanged verified HGB fits are reused.
This is a bounded alternative-bias hypothesis, not a hyperparameter/weight grid.

## Forest and linear contrasts failed; fixed weight ablation frozen

Forest study completed300 new fits, reusing300 HGB fits. RF10 adds25.33 FP per
repeat on average; its fixed HGB blend adds4. Both lose recall in a repeat and
breach the subgroup safeguard. All45 inner cutoffs,300 training/imputer records
and36 overall metric rows per assessment independently verify. Reports under
results/forest-comparison-20261016/{random_forest,forest_blend}/.

The predeclared contingent linear method completed60 deterministic fits against
cached HGB. It adds302.67 FP per repeat on average, loses recall in a repeat and
fails all primary/practical/uncertainty checks; subgroup safeguard passes.
All30 cutoffs,60 training/imputer/scaler records and36 overall metrics verify.
Report: results/linear-comparison-20261016/report.md. No retuning.

Current full suite passed228 tests before the linear test, which separately
passed. Next fixed protocol comparisons/weighting_comparison_protocol.md compares
unit-weight HGB10 and unit-weight RF10 against the SAME original balanced HGB,
with separately labelled balanced-forest reference. Earlier imbalance findings
are explicitly mixed/adverse and use a different historical objective. Only
weights change per architecture, no grid/new features.600 new fits planned;
all criteria frozen before fitting. User says continue without asking, backup only.

The unit-weight ablation completed. Both candidates fail complete acceptance:
unweighted HGB adds9.33 FP/repeat; unweighted forest adds82.67. No50,000-draw
confirmation is needed because neither passes primary checks. Do not try a
post-hoc intermediate weight or rescue an unfavorable draw result. Expanded
combined suite passes231 tests; all changed tracked research modules pass Ruff.

Next exploratory cycle addresses **which** products were purchased, a historical
family not represented by existing category shares, product counts or own-history
overlap. Only accepted event ProductID and line revenue will be used, with the
same12-month boundaries. No mutable product attributes or ambiguous archive data.
The fixed-pair SKU screen is prespecified in the existing persistent investigation
document before reading SKU/outcome differences; it is not automatic feature
selection. Any predictive vocabulary must be selected unsupervised within train.


## Historical product identity cycle — 2026-10-17 (in progress)

The no-fit SKU screen reproduces all2,750 annual/latest profiles independently
from accepted lines. Median matched TV near1 means different purchases remain
hidden by ten-feature proximity, not that identity necessarily predicts churn.
Two latest-order flags across4 snapshots motivate a bounded richer representation;
all132 IDs/scopes/k values, including negative findings, remain saved. Specific
flagged IDs are NOT chosen for prediction. Boundary/reconciliation/missing-case/
neighbor-presence/training-vocabulary tests pass4/4. The raw-source verifier's
first run used a snapshot reader on order data and failed; corrected to OrderDate
without changing source or model output, then all2,750 profiles passed.

Frozen comparisons/product_mix_protocol.md precedes300 new HGB fits. One fixed
cadence+SKU candidate uses at most20 products supported by20 distinct TRAINING
resellers, annual/latest shares+other, vocab learned inside every inner/outer
training partition. Original balanced HGB remains primary comparator; cadence
secondary. Same natural populations, grouped301/919/1201 partitions and inner
objective; unchanged5%/recall/subgroup rule, tail.05/46 and conditional50k-draw
confirmation safeguard. No grid or mutable/ambiguous source fields. Running
results/product-mix-comparison-20261017. No acceptance claim before full audit.

During product fitting, a no-outcome source follow-up checks whether shipping
service delays provide a distinct available signal. All1,886 accepted pre-April
headers have ShipDate=OrderDate+7 days with no missing values. The source duration
is constant, so no service-delay feature is fitted. Exact inputs and replay code
are in results/shipping-timestamp-audit-20261017/. System Python lacked pandas;
the recorded check ran successfully using the recreated .venv. Full combined
suite, including4 new SKU safeguards, passes235 tests in50.31s. Later changes are
no-refit verification/reporting and documentation only.

## Product cycle failed; further no-fit checks and fixed bagging trial

Product candidate completed300 fits: TP/FP201/506,202/506,199/491 versus unchanged
baseline190/498,195/551,192/505. Recall preserved each repeat; mean17 fewer FP
(3.28%), but eight MORE FP in repeat301, practical margin/uncertainty fail.
Recall interval[-1.31,+9.46]pp; FP reduction[-12.52,+47.67] at tail.05/46. No50k
confirmation is warranted. Independent300-vocabulary/30-cutoff/36-metric audit
passes; complete report includes secondary cadence reference and all subgroups.
Mean fold AP.5718 vs.5009 is descriptive, not acceptance. No vocabulary retuning.

Extended history no-fit screen fails the predicted stable overdue direction;
only9 cases gain observable cadence. CV differences are missing/neighbor-sensitive;
no feature selected. Independent1,375-case check and21,777-row shared-helper parity
pass; two new boundary tests pass. Exact purchase-fact signatures leave only one
mixed pair, but historical salesperson differs even there. Do NOT claim full
source indistinguishability, irreducibility or learnability from unique profiles.
Two collision tests pass. Full reports preserve the defined signature scope.

Product transfer diagnosis shows no universal recall-transfer collapse; score-
only/cutoff-only counterfactuals are nonadditive. Both historical transfer tables
reproduce to1e-12 after generalizing the existing utility. Seed spreads are0 in
all900 cadence/SKU fit records: five fixed seeds add no training-sample diversity.
That verified limitation motivates ONE fixed reseller-bootstrap5-member trial,
not a calibration or bag-count grid. Protocol comparisons/product_bagging_protocol.md
frozen before fits. Same original baseline, natural validation rows,301/919/1201
splits and inner objective, same criteria, tail.05/48, conditional50k safeguard.
Vocab is learned from distinct sampled TRAINING resellers; snapshots/multiplicity
travel together. Six product tests pass, including cluster resampling and repeated
case support invariance. Running results/product-bagging-comparison-20261017.

## Reseller bagging failed; coverage-motivated projection frozen

The fixed5-replica bootstrap finishes300 fits. TP/FP199/507,204/520,205/512;
mean5 fewer FP(0.97%), but additional FP in repeats301/1201. Recall point checks
pass; both uncertainty checks fail (recall[-0.17,+10.48]pp; FP[-24.93,+35.45],
tail.05/48). No50k confirmation and no bag-count/fraction tuning. Independent
resampling/vocabulary/training-hash/30-cutoff/36-metric audit passes. Complete
report includes unbagged SKU/cadence secondary references; original baseline
remains primary. Full combined suite passes243 tests in50.55s before latent tests.

Unbagged fixed-vocabulary held-out coverage is21.00% annual /24.54% latest mean,
4.78%/3.37% median; most value collapses into other. FP and TP both have low
coverage, so this is not proof of error cause. All4,125 repeated cases/scope are
saved (not independent cases), plus activity/snapshot/decision summaries in
results/product-vocabulary-coverage-20261017. This motivates a different compact
representation rather than cap/support tuning: all training-observed SKUs,
centered unscaled5-component annual-share PCA, applied to annual/latest profiles
plus unseen mass (12 added inputs versus42 direct SKU inputs).

comparisons/latent_mix_protocol.md frozen before fitting ONE unbagged cadence+
projection candidate. Same300 new HGB fits, original primary baseline, folds,
inner objective, weights/native missing/7leaves/L2=1; no component-count grid or
outcome-selected IDs. Same operational criteria, tail.05/50, conditional50k sign
confirmation. Projection is refit within every inner/outer TRAINING partition.
The2 initial training-isolation/unseen-mass/synthetic-HGB checks pass; a third
insufficient-dimension boundary check is added. Running
results/latent-mix-comparison-20261017, session21165. Verifier checks training
means, PCA orthonormality/eigen-equations/variance, vocab and expanded input hashes
without fitting a model. Root feature enumeration groups PCs before unseen shares;
actual model column order interleaves unseen by scope and is explicitly recorded
in every identity. This is metadata enumeration order, not a changed method;
preserve both rather than silently rewriting the frozen record.

## Projection failed; exact-catch ranking audit and sample-size diagnostic

Latent5-PC candidate completes300 fits: TP/FP197/534,202/544,204/523. Recall point
checks pass, but mean15.67 MORE FP(3.02%), repeat FP/practical/both uncertainty
fail. No50k confirmation or component-count tuning. Independent300 PCA/vocab/
training-hash,30-cutoff,36-metric audit passes; complete report includes reference
procedures, PCA eigen/variance checks and all subgroups. Mean training variance
captured26.74% outer /28.08% inner; compression loses substantial information,
not proof all product identities lack signal. No operational model changes.
Full suite246 passed in50.85s before the two next diagnostic tests.

Fixed exact-baseline-catch ranking audit finds directSKU lower oracle FP than
baseline ORACLE FP in12/15 outer folds (three worse); cadence12 better/one worse/
two tied; bagged10/5; latent11/4. Means9.20/9.73/10.73/7.67 fewer FP/fold. Baseline
cutoff slack is separately accounted for, preventing inflated ranking claims.
All480 oracle points reproduce by exhaustive tied-threshold enumeration; two
new tie/slack tests pass. These outcome-derived points are descriptive and never
become cutoffs. Results: results/matched-rank-burden-20261017/.

Training-size hypothesis remains unresolved, not a diagnosed cause. Frozen
comparisons/training_size_probe_protocol.md fits two fixed nested reseller subsets
(50/75%) versus reused100% directSKU fits on each existing outer fold (150 new
fits). Train vocab/weights refit inside subsets, all reseller snapshots included,
validation natural/unchanged. Reused FULL-training cutoff is deliberately only
a score-sensitivity probe, NOT a fair nested operational comparison or candidate
acceptance. Check monotonic AP/oracle burden in every repeat, report reversals,
never choose a reduced-data winner. Membership and vocabulary change too; no
causal sample-size claim. Two new cluster/monotone-mapping tests pass. Running
results/training-size-probe-20261017, session69243. No new cutoff/October access.

## Size probe failed consistency; crossfit alignment/raw control frozen

The fixed size probe completes150 fits. Declared AP monotonicity passes301/919,
fails1201; oracle-burden monotonicity passes only919. Do NOT diagnose insufficient
data or select75% as a winner from fixed full-training-cutoff results. Independent
150 training/subset/vocab hashes,45 metrics and12,375 predictions verify. Full
suite250 passed in50.82s. Median raw scores rise with fraction for both classes
in all3 mean-fold summaries, a post-hoc observation with membership/vocab confounds.

This motivates a separately frozen score-alignment hypothesis, not a causal claim:
comparisons/crossfit_percentile_protocol.md. Within each inner/outer TRAINING set,
split2 grouped stratified halves seed42. Fit fixed cadence12 HGB7/L2=1 on each
half, predict complementary reference features out of fit; reference labels do
not fit mapping. Interpolate empirical midpoint reference percentiles; preserve
ties/endpoints and average10 components (2 halves×5 original seeds). The exact
same component RAW mean is a second prespecified control candidate, both assessed
against unchanged original balanced HGB. This distinguishes mapping from changed
fitting samples/ensembling. No fractions/reference rules/seeds searched. Symmetric
halves are for independent reference construction, not retuned bootstrap fraction.
Same301/919/1201 outer, saved3 inner, natural validation rows/objective, criteria,
tail.05/54 and50k confirmation IF ALL primary checks pass.600 shared fits planned.
Two boundary/tie and synthetic-HGB/reference-isolation tests pass. Component
reference/raw/mapped validation scores and keys are saved for independent audit.
Running results/crossfit-percentile-20261017, tool session57737. Raw-versus-mapped
operational results are not yet inspected. No accepted or protected test claim.

## Crossfit alignment/control completed — neither accepted

Shared600 component fits finish. Raw TP/FP193/470,202/521,199/493: recall and fewer
FP pass each repeat, mean23.33 fewerFP(4.50%), but practical and both uncertainty
fail. Percentile TP/FP189/454,199/490,196/456: mean51.33 fewerFP(9.91%), positive FP
uncertainty lower21.50, but repeat301 loses ONE baseline-caught churner (189 vs190)
and recall uncertainty[-2.83,+5.49]pp crosses0. Strict baseline recall criterion
is not waived. No50k confirmation, reference/half/seed tuning or post-hoc cutoff.
Independent600 component/reference/training identities,45 cutoffs and36 metrics
per child verify; both complete reports exist under results/crossfit-percentile-
20261017/{crossfit_raw,crossfit_percentile}/. Full combined suite252 passed in53.29s.

Next no-fit screen returns to residual historical information of the simpler
fixed cadence model. Original latest paid unit revenue contrasts were lower for
persistent recent false alerts and inactive continuing buyers; this accepted
subtotal/quantity descriptor is distinct from unverified archived cost/profit.
Prior latest-line/annual-diversity trials failed, explicitly reviewed; no automatic
repeat of those features. Frozen residual screen in the existing investigation
protocol rebuilds cadence3-repeat alert roles, same missingness-aware snapshot
matching and all17 descriptors/k1/3/5. Results currently being produced in
results/cadence-residual-history-20261017, session41074. No model/cutoff selection.

## Cadence residual unit-revenue hypothesis and one frozen numeric feature

Residual screen completes:360 persistent recent false alerts,3 consistent misses
(allApril),63 inactive continuing buyers. K3 latest paid-unit-revenue medians are
lower for recent FP in all4 snapshots; nearest means negative in all4, January
mean reverses at k3/5. Inactive direction reverses twice, and3 missed cases cannot
support a reliable contrast. All17 descriptors/k1/3/5 retained, no p-values.
This is partial motivation for a compact feature, not proof of mechanism.

Frozen comparisons/cadence_unit_revenue_protocol.md precedes ONE13-feature
cadence+last_order_unit_revenue trial (300 new fits, cached original baseline,
cadence secondary). Earlier line-count/diversity failures are explicitly reviewed;
this average accepted paid amount per accepted unit is different and never means
archive cost/profit. Same fixed parameters/seeds/outer/inner populations/objective,
raw feature with no transform/grid. Criteria unchanged, tail.05/56, conditional
50k confirmation only if ALL primary checks pass. A small shared numeric-feature
runner avoids duplicating the nested fitting workflow; its synthetic fold smoke
passes baseline parity, recorded20+20 fit identities, group isolation and finite
scores. Running results/cadence-unit-revenue-20261017, session76025. Existing
comparisons.verify_basket_breadth independently supports this fixed numeric-HGB
layout (40 cached+new identities/fold), so no new duplicate verifier is required.


## Latest numeric feature completed; evidence boundary

300 new +300 cached baseline fits, no operational changes. Cadence-unit-revenue
TP/FP199/471,207/511,203/488 vs190/498,195/551,192/505. All repeat recall/FP points,
5% practical margin, activity/snapshot safeguard and recall uncertainty PASS.
Mean recall90.72→95.75%, FP518→490 (28 fewer,5.41%). FP conditional interval
[-2.5122,+60.2277] crosses0 at frozen tail.05/56; recall[+.46996,+10.3751]pp.
**Complete acceptance FAILS.** No50k-draw confirmation (primary did not fully pass),
criterion relaxation or near-result parameter/cutoff tuning. Secondary cadence
TP/FP196/470,207/527,202/489, so added descriptor not uniformly better there either.
Independent reused fixed-numeric verifier checks all600 cached/new training hashes,
15 partitions,30 cutoffs,36 metrics and8250 paired rows; passed. Full report:
results/cadence-unit-revenue-20261017/report.md. Full suite253 passed in60.40s;
all latest modules/tests Ruff clean and git diff --check clean. Task-only known
matplotlib temporary cache removed; all research evidence/user files preserved.

No accepted development replacement has been established. Source-only inquiry
is bounded by the user's confirmation that CompanyX.bak is the only source: it
contains no verified requested contact/service/payment event series or order
version/cancellation transitions; archive ActualCost semantics remain unresolved.
The latest fixed method has promising points, not sufficient frozen evidence.
No new source-backed remedy is currently motivated by these diagnoses. This is
not an impossibility claim about all algorithms/history representations. Further
near-margin tweaking/repeated draws on the same422 resellers would not supply
new independent confirmation. The next evidence-bearing step is additional
independent reseller outcomes with verified prior histories, or an appropriately
untouched cohort whose roles/method are frozen and explicitly authorized before
outcome access. An eligible cohort MAY exist inside the backup; none is assumed
approved or queried as a new independent test. October is already inspected.
Do not ask again for unavailable source owners/another source; user said backup
only and continue without questions. No protected evaluation, full-data model
promotion, deployment, commit/push/merge or changes to the operational bundle.

## 2026-10-17 — explicitly authorized latest-candidate October evaluation

Observed problem: latest13-feature cadence/unit-revenue candidate improved grouped
development points but failed FP uncertainty acceptance. User explicitly requested
“use that candidate and evaluate on the test set,” identifying October as the test
and excluding test-based development. No further approval or source-owner question.

Before final fitting, froze comparisons/unit_revenue_october_protocol.md. Candidate
specification copied exactly from results/cadence-unit-revenue-20261017/protocol.json;
original10 baseline has identical development cutoff objective. Historical0.06
reported separately, never weakened. Full1375-row training,212churners,422resellers;
3 grouped OOF folds seed78; precision maximum subject recall>90%,ties recall/cutoff;
5 fixed seeds,balanced training weights,native missingness. Candidate cutoff
0.18634604988648462; matched baseline0.0067626906041783315. Forty fits total;
no October labels enter fitting or selection. Sources reconcile all13 development
and10 October columns. Tracked stores bytes differ solely byLF/CRLF and reproduce
the original recorded hash under CRLF; no tracked input or old output changed.

Saved source/code hashes,models,training identities,inner assignments,OOF scores,
cutoffs and1476 unlabeled predictions in new
results/unit-revenue-october-evaluation-20261017 before reading October labels.
Explicit evaluation joins492 exact keys/46churners. Candidate45TP/194FP/1FN/252TN,
recall97.83%,precision18.83%. Historical43TP/108FP/3FN/338TN,93.48%/28.48%.
Matched-policy original10:46TP/183FP/0FN/263TN,100%/20.09%. Objective FAILS against
both. Candidate adds2catches but86FP against historical; loses1catch/adds11FP
against matched policy. All candidate misses/additionalFP are recent purchasers;
all35inactive churners/10inactive continuing accounts alerted by all3strategies.
OverallAP rises but recent-purchaserAP falls; neither rescues operational failure.

Paired5000-reseller95% bootstrap seed2026: historical recall difference[0,11.36]pp,
FPreduction[-105,-67]; matched recall[-7.32,0]pp,FPreduction[-29,+7]. No empty
churn draws; intervals omit training/cutoff/source/adaptive-research uncertainty.
Source completeness/order-time availability remain assumptions. Supplied labels
are prior accepted-order reconstructions; historical scores are a reproduction.
No newly untouched-cohort claim,no test-driven tuning,no promotion/deployment.

Execution checks/issues: focused15tests passed before fitting. Independent
verification initially rejected in-memory estimator joblib hashes after reloading;
serialization changes these hashes for10/10models, while serialized file SHA,
parameters and exact scores pass. Evaluation saved cases,metrics,paired intervals
then failed Markdown rendering because tabulate was absent. Preserved these outputs;
comparisons/verify_unit_revenue_october.py independently reproduced40training
identities,3group-isolated folds,2cutoffs,10saved-model predictions,9confusion rows,
and all paired intervals. It writes completed_report/report.md and manifest,
checking the exact archived as-run entrypoint. Current runner removes that optional
renderer dependency; added an end-to-end synthetic report/overwrite-guard check.

Learning: candidate does not achieve fewer false alerts with preserved baseline
recall on October. The requested evaluation is complete. Keep the frozen method,
all unfavorable results and original operational model. No next remedy may be
selected using these test outcomes. New code: dedicated prepare/evaluate runner,
no-fit verifier,protocol and focused tests. See current continuation checkpoint.

Final verification: full combined suite **258 passed in61.51s** using the
recorded combined pytest command; new runner/verifier/tests pass Ruff and
`git diff --check`. New matched-policy baseline scores reproduce saved historical
baseline scores with maximum absolute difference5.421e-20; only their cutoff
policies differ. No active process remains and no disposable files were retained.

## 2026-10-17 — development-only audit of validation failure

User asked what went wrong and explicitly prohibited touching the final test.
Read only latest development reports,manifests,predictions,inner cutoffs,split
memberships,relevant code and development research history. No test input,
label,prediction,bundle or test report accessed. New no-fit entrypoint:
comparisons/development_validation_audit.py; results/development-validation-audit-20261017.
No models fitted,no cutoff selected,no next candidate chosen.

Confirmed: reseller grouping does not enforce calendar order/matured labels;
all60 fold/snapshot cells contain unresolved-at-prediction training labels.
Average development gain hides April FP increases33/24/30 in all3repeats.
The old subgroup acceptance bounded recall loss,not period/activity FP increases.
Candidate mean inner→outer recall91.08→95.75%,FPR38.86→42.11% indicates operating
transfer variation; different models/populations prevent causal attribution.
Repeated research used same422resellers; candidate hypothesis derived from
previous outer errors,nesting doesn't cover that meta-selection. Five seeds
identical. Added unit-revenue beyond cadence yields only1.33TP/5.33FP improvement
on average and is not uniformly better. Recent-purchase prevalence varies4.18–
17.20% even in development. Candidate already failed uncertainty acceptance.
No new evidence identifies an implementation/feature-leak bug as causal.

Proposed correction is validation design/evidence discipline before more models,
not an October-informed feature/cutoff remedy. Grouped and matured-label temporal
estimands must be explicit. Naively excluding ALL future-cohort resellers gives
January76/76 positive training rows; cannot silently switch to that biased setup.
No claim all temporal/grouped designs are impossible. No acceptance criteria
changed retrospectively. See report and docs/debugging.md for facts/limitations.

Independent no-fit recount checks12 snapshot/repeat comparisons,60 calendar cells,
and120 inner/outer/activity rows; source manifests pass. Ruff passes. Only new
analysis script/report and documentation were added; no shared model behavior
changed and no training/test evaluation was repeated.

## 2026-10-17 — correction after reviewing prior temporal selection

User noted temporal validation already gave weaker results. Checked existing
controlled comparison report,code,frozen selections,six aggregate metric rows and
summary. GroupKFold beats temporal on same April2013 population at all3budgets:
17/24/34 versus9/21/32 catches,AP0.4183 versus0.2813. Formal winner inconclusive
(CI crosses0); selected feature/imbalance procedures differ and one temporal
period/partial early history limit inference. This confirms temporal was tried;
our previous explanation overclaimed the grouped time-mixing limitation as a
cause and implied an untried temporal fix. Corrected docs/debugging.md accordingly.
Saved controlled comparison explicitly identifies April DEVELOPMENT evaluation;
inspected October final protocols/aggregate reports are grouped-selected,not a
verified temporal-selected October test. No case-level test features,labels,
predictions or model bundle accessed,no fitting,no new evaluation or tuning.

## 2026-10-17 — execute development preparation plan

User authorized implementation of the six-step preparation plan and prohibited
changes to AGENTS.md. Preserved all existing local work. No real candidate was
trained, no new feature or cutoff chosen, and no final-test dataset, predictions,
labels or model bundle accessed. Tests use synthetic fixtures; the no-fit review
uses only results/cadence-unit-revenue-20261017 development artifacts.

New code: comparisons/development_checks.py validates approved snapshots, unique
case keys, labels, predictor exclusions, native missingness, reseller isolation,
optional temporal label maturity and supplied feature times. Shared partition and
fixed numeric-feature fitting paths call these guards before candidate fitting.
comparisons/development_reporting.py adds paired repeat/fold/snapshot/activity and
intersection metrics, sorted worst-slice flags, undefined-rate handling and saved-
cutoff transfer tables including component training row/reseller ranges.
comparisons/review_development_study.py reads the declared fixed-numeric development
layout only and verifies canonical counts, manifests, split memberships and matched
predictions; it performs no fitting, model loading or threshold selection.

Shared assessment writes operating_metrics.csv, paired_operating_metrics.csv and
worst_slices.csv. An explicit optional period_limits setting applies prospective
per-repeat snapshot/activity recall/FP safeguards; fixed-feature run protocols
persist it. Limits are validated before fitting. Omission preserves legacy
acceptance for old callers. New experiments must freeze/pass limits in advance;
proposed research defaults are at most five percentage points recall loss and no
additional FP per period/activity cell, alongside unchanged overall baseline-recall
preservation, strict FP reduction, practical and uncertainty requirements. Undefined
recall does not silently pass. No historical candidate is rejudged under new limits.

New preparation protocol: comparisons/development_research_protocol.md. Updated
existing research rules, pipeline guide, metric guide and README. Temporal selection
is not asserted superior: it was already tried and weaker on points in the controlled
comparison. Repeated validation remains legitimate development; no improvement is
claimed by a reporting label. Remaining next-fit choices (candidate, exact splits,
uncertainty family/tail and numerical period limits) must be frozen before fitting.
No new candidate is defined or authorized by this preparation document.

Evidence: initial no-fit review results/development-preparation-review-20261017 is
retained with its as-run identities. Final review after implementation completion:
results/development-preparation-review-20261017-verified/report.md. Independent
checks reproduce252 historical metric rows and exact original acceptance decision,
verify450 saved-cutoff transfer rows and all three adverse April FP comparisons.
An initial strict decision comparison used pandas default float parsing and differed
only at about4e-17; using the established round_trip parser reproduces exact values.
No source/output was rewritten to satisfy that check. New end-to-end synthetic
review tests forbid fitting and cutoff selection and verify overwrite rejection.
Focused preparation suite20 passed; final combined suite result is recorded below.

No AGENTS.md edits, old-artifact replacement, resets, commits, pushes, merges,
model promotion or deployment. No further model work is started in this task.

Final verification: **278 tests passed in64.98s** with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests debug comparisons/test_grouped_precision_recall.py comparisons/test_grouped_ablation.py comparisons/test_evaluate_frozen_october.py`.
Focused preparation tests20 passed. Changed modules/tests pass Ruff;
`git diff --check` passes. Final review output manifests and current implementation
hashes pass; no disposable task files or running processes remain.

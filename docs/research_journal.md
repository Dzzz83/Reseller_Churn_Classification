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

## 2026-10-18 — autonomous development continuation, reconciliation and cycle 1

Created research/development-recall-alerts-20261018 from db52abd in the clean
primary checkout. experimental_research at c047741 has user-owned ablation changes
and remains untouched. Recent research lives in pipeline_preparation plus ignored
results, not just that experimental checkout. Reviewed the four requested saved
development reports and the ablation protocol/results. No prior candidate passes
complete acceptance. Ablation uses a different operating policy; no direct headline
comparison is valid. Historical old feature name remains preserved in research.

Mixed historical documentation unexpectedly displayed protected aggregate results;
this is a boundary incident, not new evidence for selection. Those contents are
excluded from this task's hypotheses and decisions. Subsequent reads are scoped
to explicit development evidence. Prior test inspection means no independent test
claim is available. No test artifact, case, prediction or label file was opened.

New hypothesis derives from development transfer and the failed raw/percentile
studies: retain classifier/reference-calibrator pairs rather than map scores then
refit a different classifier. Freeze comparisons/retained_calibration_protocol.md
before any calibrator fitting. Saved raw component/reference scores permit zero
HGB refits. Supervised regularized logistic maps differ materially from previous
unsupervised percentile maps; unchanged raw ensemble is the controlled reference.
Prospective period/activity/intersection safeguards and alpha-spending are fixed.
Next: implement isolated reconstruction, synthetic checks, execute, verify all
results independently and diagnose failures before deciding another cycle.

Cycle1 completed: retained_logistic TP/FP301=189/429,919=198/464,1201=193/431;
baseline190/498,195/551,192/505. Mean76.67 fewerFP(14.80%) but repeat301 loses
one catch; October2012 and April period/intersection gates fail. Recall interval
[-.0428727,+.0534454],FPreduction[43.6663,113] at tail.00025. Complete acceptance
FAILS. No50000-draw confirmation or calibrator tuning. Zero new HGB fits,600
calibrators; identical seeds do not add evidence. Independent audit checks600
identities,540 operating rows,30 cuts,5000 bootstrap draws and logistic gradients
(max9.52e-9),all pass; no negative slopes. Results/retained-calibration-20261018.
A repeated manifest write caused a self-checksum defect; original manifest/as-run
source preserved, separate completion_manifest.json records finalized assessment.
Verifier uses it and checks archived pre-fit code hash. No scores or decisions
were changed to repair metadata. Nine focused/synthetic tests pass.

No-fit next diagnosis results/retained-ranking-audit-20261018 exhaustively verifies
12 tied period burden points at baseline catches. October2012 losses29/27FP in
301/1201; April loss8FP in301. Inner/outer average operating mismatch is much
smaller, yet acceptance fails: no justification for repeated calibration tuning.
Cycle2 frozen in comparisons/smooth_cadence_protocol.md: a single smooth additive
cadence learner,not the previous linear10 or a hyperparameter search. Next run
focused preprocessing/isolation/extrapolation checks,60fits,independent reconstruction.

Cycle2 completed,results/smooth-cadence-20261018-02:60 deterministic spline/logistic
fits; baseline cached. Mean56 MORE false alerts(10.81%); overall recall point
preserved each repeat,but uncertainty/period/practical/FP requirements fail.
Complete acceptance FAILS. Independent60training identities,medians,quantile
knots,scales,coefficients/held scores,30cuts,540operating rows and all42period
acceptance cells/5000draws reproduce. Maximum logistic gradient1.34e-7.
Focused two synthetic checks pass,including nested cached-baseline parity.
First results/smooth-cadence-20261018 interrupted after one fit due duplicate
metadata keyword; preserved,corrected caller contract,added end-to-end test,
no scientific parameters changed. Initial verification test also exposed constant
spline knot reconstruction semantics; scipy construct_fast reproduces sklearn.

Cycle3 observation: counts of snapshots per reseller are strongly associated with
development labels. Training-only inverse-history/class-normalized weighting is
not covered by prior unit weights or whole-reseller bootstrap product trial.
Freeze comparisons/reseller_weighting_protocol.md. Original10HGB unchanged except
training weights,300fits,strict same safeguards,new alpha spend. No history count
becomes a predictor and no evaluation row is removed/reweighted. Equal reseller
versus row influence are different fitting choices,not proven bias correction.

Cycle3 completed,results/reseller-weighting-20261018:300newHGB fits,300cached
baseline identities;31moreFP(5.98%) on average,recall preservation fails. Complete
acceptance FAILS. Independent300weight identities,class totals/parameters,
15splits/30cuts/36overall metrics/8250predictions and5000draw/full-decision audit
pass. Six focused/synthetic checks passed before execution. Saved-cutoff review
results/reseller-weighting-transfer-20261018 has450transfer rows,actual component
sizes; no fitting or new cutoff. Hypothesis of useful account-weight correction
is unsupported in this fixed trial;no exponent or weight retuning.

Cycle4 was frozen contingently before cycle3 outcomes:two activity experts with
original10features/architecture/full-partition weights. No case removed,no routing
threshold search. First run results/activity-experts-20261018 hit sklearn1.9.1's
all-missing-column binning error before any expert fit/prediction. Synthetic60row
reproduction and local source establish cause. Version2 implementation amendment
omits only entirely missing FITTING columns,records actual features and retains
all10input fields/hashes. No observed feature information removed,other NaNs
native. Both frozen versions/source snapshots preserved;no outcomes motivated
repair. Two focused routed/all-missing/empty-held/missing-class tests pass.
Results/activity-experts-20261018-02 now executing600expert fits,unchanged gates.

Development lineage reconciliation results/development-lineage-reconciliation-
20261018 proves ablation,research and renamed operational preparation have exactly
the same1375keys/212labels/422resellers,ten numeric features and missing values
under explicit name mapping. Byte hashes intentionally differ across serialization
and schema. No raw future/test rows were read for reconciliation.

Cycle4 completed,results/activity-experts-20261018-02:600expert fits with only
training-all-missing columns omitted,300cached baseline fits;21fewerFP(4.05%),
adds2FP in301,loses6catches in1201. Complete acceptance FAILS. Independent600
expert/subset/global-weight identities,routed scores,30cuts,540operating rows,
5000draw/full-decision audit pass.52 consolidated synthetic/boundary tests pass.
Transfer450rows and twelve independently enumerated tied period burden points
saved separately. No50000confirmation,route/parameter/cutoff tuning or promotion.

No-fit continuation-complementarity-20261018 finds bidirectional catches7/12,
7/4,16/9 and meanwithin-foldmidrankrho.85957 between calibration and routedexperts.
These are different fitted procedures,not five identical seed evidence. Experts
improve April ranking in all repeats;calibration gains mainly elsewhere. This
provides a specific new ensemble hypothesis rather than revising a near-failed
cutoff. Freeze comparisons/continuation_blend_protocol.md forONE50/50 mean using
training-only analytic odds normalization of globally balanced expert scores.
Report correctedexpert as ineligible mechanism control. Zero newHGB/calibrator
fits. Same strict acceptance;cycle5alpha ledger continues. Next verify correction
boundaries/isolation,execute saved-score comparison,independently audit every result.

Cycle5 completed,results/continuation-blend-20261018:zeroHGB/calibratorfits,
TP/FP188/435,199/475,193/436 versus190/498,195/551,192/505. Mean69.33fewerFP
(13.38%),but repeat301 loses2catches;April adds8/6/5FP and period/intersection
limits fail. Recall interval[-.0449141,+.0543567],FPreduction[40.0521,107.1720].
Complete acceptance FAILS. Independent120score-stage constructions,45cutoffs,
1080candidate/control operating rows and5000draw/full-decision audit pass.
Mechanism control not eligible and did not resolve expert repeat losses.
Seven focused prior-correction/pairing/leakage tests pass. No blend-weight tuning.

Cycle6 frozen:period/class-balanced ORIGINAL10HGB loss changes ONLY fitting weights,
not models/features/cutoffpolicy/splitter. This targets persistent period tradeoffs,
which inverse-reseller weights did not address. Equal8period/class cells is
simple fixed reweighting,not a claimed implementation of groupDRO or forecast fix.
See comparisons/period_weighting_protocol.md. Same strict complete acceptance,
next alpha spend,300fits after focused checks. No protected outcomes selected it.

Before cycle6 results,freeze contingent cycle7 only if cycle6 fails:one RBFmargin
learner on cadence12. Failed linear/additive boundaries versus tree period-ranking
losses motivate a smooth multivariate interaction function,not a kernel/C/gamma
search. Explicit probability=False avoids libsvm's ungrouped internal probability
CV;expit margins remain non-probabilistic scores. See rbf_cadence_protocol.md.
No model/feature/cutoff choice comes from protected-test evidence. Next retain
all failed decisions and execute only the justified frozen contingent procedure.

Cycle6 completed,results/period-weighting-20261018:original10HGB equal8cell
loss,300newfits,TP/FP192/511,191/545,192/560. Mean20.67MOREFP(3.99%),repeat919
loses4catches;complete acceptance FAILS. Independent300weight/cell identities,
15partitions/30cuts/36overallmetrics/8250predictions and5000draw/full-decision
audit pass. Six focused weighting tests pass. No weight/class/period retuning.
Cycle7 contingent criterion is met;RBFcadence60fit procedure is executing after
its synthetic kernel/transform/dual-bound/missing/extrapolation check passed.
Explicit probability=False emits a sklearn1.9 future deprecation warning;it
is intentional to avoid ungrouped internal probabilityCV,not an execution failure.

Cycle7 completed,results/rbf-cadence-20261018:60deterministic fits,no probability
calibration,mean141.67MOREfalsealerts(27.35%) despite overall repeat recall
preservation. Complete acceptance FAILS. Independent60support/preprocessing/
training identities,kernel-margin reconstruction/dual bounds,30cuts,540operating
rows and5000draw/full-decision audit pass. No C/gamma/scale/cutoff tuning.
RBFtransfer450rows saved separately. Generic transfer reviewer now labels
single deterministic estimators explicitly rather than requiring a seed field;
previous exact reviewer source is archived. No models were refitted for reviews.

Seven completed cycles are consolidated in results/development-continuation-
20261018:summary/repeatcounts/uncertainty/source-code snapshot/runtime/identities.
All failed decisions remain failed. Completed real fit slots:1200newHGB,
60smooth-additive,60SVM,600reference logistic calibrators;one additional first-run
spline fit was discarded after metadata failure. Synthetic checks are separate;
identical five-seed predictions do not add robustness. Prior original/crossfit
model fits were reused,not rerun to read results. No50000draw rescue was allowed.

Next investigation before any fit:read primary method guidance on monotonic
shape constraints and establish whether recency/cadence directions are defensible
conditional assumptions for this purchasing data. Saved-score associations are
not partial dependence or causal proof. No monotonic candidate,parameters,margin
or cutoff is frozen yet;do not infer authorization for test access or tune the
seven failed methods. The objective remains unmet and research remains active.

## 2026-10-18 — shape investigation and cycle 8 freeze

The previous goal turn made verified progress (seven complete comparisons), not a
wait or no-progress turn. Rechecked current branch/status, AGENTS, and the clean
standalone development checkpoint; preserved all existing research changes.

No-fit recency-shape diagnosis: April recent-buyer churn rates reverse the simple
marginal recency trend; exact annual-count strata are sparse and do not establish
conditional partial dependence. Primary BG/NBD P(alive) conditions on frequency,
observation age, and a specific dropout process; it is not the six-month label.
No hard monotonic constraint is selected merely from that literature or a marginal
association. results/recency-shape-diagnosis-20261018 independently reconciles four
half-open historical quarter counts to all 1,375 annual/recent counts, using saved
accepted pre-snapshot trajectories only. No model or cutoff was fitted.

Historical-quarter diagnosis reused unchanged original and cadence-residual
matches, all four count/share descriptors, and all k=1/3/5 summaries. Earliest
quarter share has small mostly negative original recent-FP contrasts (k1/k5 all
periods; k3 tiny January reversal), mostly zero medians. Cadence residual direction
reverses in April. Results/quarter-history-diagnosis-20261018 and
quarter-cadence-residual-diagnosis-20261018 preserve every adverse contrast. No
seasonality or causal claim. This feature choice is explicitly exploratory after
eight descriptor summaries, not nested human hypothesis selection.

Freeze cycle 8: original HGB10 + early_quarter_order_share, all original parameters
and weights unchanged. This is a controlled single-feature trial, not addition to
the cadence parent whose residual evidence is weaker. comparisons/early_quarter_
share_protocol.md owns exact formula, same splits/objective, unchanged strict
operating/practical/subgroup/uncertainty gates and cycle8 alpha spend.

Numerical-readiness issue: later alpha tails imply far below one expected tail
observation in 5,000 draws. Historical seven decisions already failed independent
point/period gates and remain failed. A new ADDITIONAL simulation-resolution gate
is frozen before cycle8 fitting: after passing original5k and50k gates, require
fixed10,240,000 paired draws plus exact one-sided binomial bounds on bootstrap
failure probabilities. This can only reject, never rescue or loosen a failed
criterion. It addresses Monte Carlo error, not adaptive research/model coverage.
Next implement focused source/feature checks and independent reconstruction, then
execute the single frozen comparison if correctness passes.

Cycle8 completed, results/early-quarter-share-20261018: 300 original-parameter
HGB fits plus unchanged cached baseline. Recall point preservation passes every
repeat, but mean FP increase is 6 (1.16%) and period/intersection safeguards fail.
Complete acceptance FAILS. Independent calendar audit checks all 5,500 quarter
counts/1,375 features; nested synthetic feature and simulation tests pass (9 tests).
No-fit split/training-hash/cutoff/count, feature/lifecycle, and full acceptance
reconstructions pass. The frozen extra Monte Carlo gate is correctly NOT RUN:
failed 5,000-draw primary results cannot receive 50,000 or 10.24-million-draw rescue.
No quarter, transform, parameter or cutoff retuning follows this failure.

Next distinct hypothesis: train-only guarded cutoff selection with an explicit
baseline fallback may prevent unsafe use of a component whose average FP improves
but period harm persists. Use only original HGB and fixed retained-logistic
component; no model pool expansion or blend-weight grid. Freeze a common inner
objective and all subgroup constraints before any outer construction. Baseline
fallback retains every outer case and is an explicit new procedure, not retroactive
reinterpretation of failed fixed components. Existing protected-data boundary and
all outer acceptance requirements remain unchanged. No new classifier fits needed.

Cycle9 completed, results/guarded-component-20261018: ZERO new fits. The frozen
training-only common cutoff constraints find no feasible retained-logistic cutoff
in any of15outer-training partitions. Every fold therefore takes the explicit
original baseline fallback. All4,125 paired candidate decisions equal baseline;
FP reduction0 and complete acceptance FAILS. This is not a safety-only success or
a development improvement. No guard was relaxed and no favorable folds were pooled.
All cutoff scans, reference-group counts and fallback reasons are retained.

The common-baseline, intact-tie, recall-boundary, undefined-group, practical-fallback
and no-extra-draw-rescue tests pass (10 tests including streamed uncertainty tests).
An independent exhaustive tied-cutoff verifier and full acceptance recount are
running. Next diagnose which training FP lower bounds and recall upper bounds
conflict, retaining all groups. That can distinguish missing discrimination from
merely choosing the wrong threshold, without retuning cutoffs on outer labels.


## Wrap-up at user request — 2026-10-18

Cycle9 independent audit PASSED:32690tied boundaries,15choices,540operatingrows.
Binding-bound diagnosis finds AprilFP lowerbounds conflict with earlierperiod
recall upperbounds. Exactintegerbootstrap diagnostics reproduced conditional
sign probabilities without changing any failed decision;15exhaustive tests pass.

Cycle10 frozen positive_floor_veto completed,zero new fits. Learnedminimum
innerOOF retained-logistic score among baseline-caught churners;unchanged baseline
cutoff,strictbelowfloorveto,tiesretained. Independent source/cutoff/floor/outer
reconstruction passed15folds,8250predictions,540operatingrows,42cells and5000draws.
TP/FP188/451,195/519,191/459 versus190/498,195/551,192/505. FP reduction8.04%
but recall lost2/0/1cases across dependent repeats;Octoberrecent recallloss6.67pp
exceeds5ppguard. Complete acceptanceFAILED. No confirmation/exact-rescue run.
The user requested finishing current work and reporting the last hour without
requiring the goal. No new cycle follows. Standalone safe checkpoint/report:
results/development-wrapup-20261018. No final data accessed in this cycle.

Final wrap-up checks: 60 focused tests passed in 16.86 seconds; one existing sklearn
SVC deprecation warning. New veto runner/verifier/tests pass Ruff format/check;
git diff --check passes. No task-created temporary files remain. Independent audits
and immutable source identities are bound in the new wrap-up manifest.


## Explicitly authorized inspected-test evaluation — 2026-10-18

User: “evaluate on the final test set.” This supersedes the earlier no-test-access
instruction for this one evaluation, without authorizing test-driven tuning.
Frozen positive-floor veto:full-development baselinecut.0067626906041783315,
componentfloor.06083151162511704,40HGB+40maps fromdevelopmentonly. Original10full
baseline/OOF reused after identity checks. Predictions checkpointed and independent
preparation audit passed before supplied outcomes. No DB query/labelconstruction.
October candidate46TP/171FP/0FN/275TN,recall100%,precision21.20%;matchedbaseline
46/183/0/263,100%,20.09%. Historical.06:43/108/3/338,93.48%,28.48%.
PairedFP reduction12(6.56%),95%conditionalinterval[6,19],recallchange[0,0].
All changes recentpurchase;inactiveunchanged. No procedure adapted after outcomes.
Development failure remains. Octoberreused,notnew independenttest.
Originaloutcomejoin failed datetime/string; recovery normalizedkeys; duplicate
reportfield failure preserved in evaluation/. Completed_evaluation is authoritative.
No refit for reportingrepairs. Independent40fitidentities/1476decisions/9metrics
reproduce. In-memoryjoblibhash changesafterserialization were synthetically
reproduced; bundleSHA plus exactcomponentpredictions used for persistedidentity.
Report:results/positive-floor-october-evaluation-20261018/summary.md.


## Candidate-specific threshold follow-up — 2026-10-18

User requested selecting the candidate's own development threshold before October
evaluation. Removed baseline-TP and baseline-cutoff constraints; maximized pooled
OOF precision subject to recall strictly above 90%, ties recall then higher cutoff.
Independent exhaustive tied-score enumeration agrees with the existing selector.
It selects the SAME cutoff,0.0067626906041783315: development recall90.09%,precision
28.72%,191TP/474FP/21FN/689TN. No numeric threshold difference or changed test
predictions. October remains46TP/171FP/0FN/275TN,100%recall/21.20%precision.
No refit, floor revision or test-based selection. Predictions checkpointed before
supplied outcomes; manifests verified. Descriptive reuse of inspected October;
full development acceptance still failed. Report:
results/positive-floor-own-threshold-october-20261018/evaluation/report.md.


## Authorized precision-floor October sweep — 2026-10-18

User requested minimum development precision20%,30%,40%,... then final evaluation.
Fixed model/veto floor; no new fits. Frozen targets20–100% in10point steps; maximize
development OOF recall with precision >=target, ties higherprecision then cutoff.
No90%recall floor in this policy. All nine cutoffs and test predictions saved
before supplied outcomes. Independent exhaustive tied-cutoff selection and27group
confusion recounts pass;two synthetic tests pass. Target30% gives test45TP/154FP,
recall97.83%,precision22.61%;40% gives40TP/60FP,86.96%,40.00%;50% gives31TP/23FP,
67.39%,57.41%. Targets80–100% select one development alert and no test alerts;
test precision undefined, explicitly flagged. No test-winner selection or deployment.
October previously inspected; descriptive reuse. Full table and undefined-rate
correction:results/precision-floor-october-20261018/report.md. Original outputs
remain immutable;audited_comparison.csv provides undefined precision display.


## Authorized all-candidate precision-floor evaluation — 2026-10-18

User requested the same precision-target evaluation for other candidates. Frozen
scope: ten recent procedures plus two earlier cadence candidates and original
baseline,13 total. Development OOF cutoffs maximize recall subject to precision
>=20%,30%,...,100%;ties higherprecision then cutoff;no recallfloor. All117targets
frozen before outcome stage. 92 feasible and25 infeasible,all reported. Same492
October resellers/46churners,inspected previously;descriptive reuse,not independent
validation. No test winner selection,model tuning,deployment,merge or push.

At40%developmentprecision target,testprecision/recall/FP: original40.00/86.96/60;
retainedcalibration38.18/91.30/68;smooth52.86/80.43/33;reseller42.71/89.13/55;
experts43.30/91.30/55;blend41.58/91.30/59;period45.16/91.30/51;RBF43.82/84.78/50;
earlyquarter39.62/91.30/64;guarded=original;veto40.00/86.96/60;oldercadence35.40/
86.96/73;unitrevenue34.26/80.43/71. These are separate predeclared policy outcomes,
not a test-selected winning method. Full20–100%grid and counts in the report.

Independent audit:117cutoff choices,128logicalfitidentities,7fullmodelbundles,
276overall/activity metricrows all reproduce. Guarded procedure selects baseline
fallback;its duplicate results are intentional. Legacycadence runtime/Windows
manifest incompatibilities handled via verified originalspec and fixed refit.
Interruptedpreparation attempts preserved;fullmodels cached,75losttransientinner
HGB predictions recovered by exactidentity-matched fits. Finalroot:
results/all-candidate-precision-october-20261018-04/report.md. All frozen
DEVELOPMENT acceptance failures remain;no independent final-validation claim.


## Missed recent-purchase warnings — development-only diagnosis, 2026-10-18

User asked for unused warnings among hardresellers. Fixed period/class40% OOF
policy,67missed recentchurn cases/66resellers;noOctoberinputs/no new fits.
Rebuilt6875historicalratios and all603matchedneighborhoods;3synthetic tests pass.
47of67have at leastoneobservedwarning,15noneobserved,5insufficientcomparisonhistory.
Smallerlastorder40/62;revenue decline23/53;longerlastgap20/51. Continuingbuyers
alsofrequentlyshowthese:67.3%,63.2%,39.5%. Main k3matchedflags do not separate
missedchurners. Continuousrevenueratio andproductvariety offer exploratory hints,
with nearest-only attenuation/reversals;related previous features failed full
acceptance. No new predictor or fix established. All18descriptors/k1,3,5/periods,
missingness/coverage/controlreuse saved. Fullreport:
results/hard-recent-warning-audit-20261018-02/report.md. Initialschemaadapter
failure preserved;no trainingrepeated. Boundeddiagnosis complete,no active fit.


## Fixed product-engagement ranking experiments — 2026-10-18

New user research objective:move churners above continuing buyers without many
more false alerts. Development-only, no further October access. Froze two arms
under same period/class HGB10 and nested301/919/1201x5/inner3seed78:annualdistinct
products;distinctproducts/annualorders. Same training-only40%precisionpolicy;
all settings,criteria andcycle11/12spends frozen before fits. Neither passes full
acceptance. Breadth APgain+.00754 reverses inone repeat;meanFP reduction3.33(1.53%)
and recall lost in1201. Density APgain+.02117,AUCgain+.02523,pointgains positive
all3repeats;AP95%interval[-.00327,.04772],AUC[.00965,.04068]. MeanTPgain8,FPaddition10.
Densitytherefore offers provisionalranking information,not practicalsuccess.
No failedprimary receivesconfirmation. Independent600modelreloads/60cutoffs/
1080operatingrows/5000operatingdrawsperarm/17rankingdrawspots pass. Root:
results/period-product-ranking-20261018. Protocol and currentasrunsnapshots saved.

Contingent follow-up frozen before fitting:personalizedrecency(original10+overdue)
under sameweights/parameters/policy;oldcontrolled f01vsf00 APgainpositiveall3old
repeats motivates it. Cycle13. Separate densityattributioncontrol annualordercount
alone,cycle14,justified by provisionalratio signal and predeclared attribution
limit,independent ofpendingrecencyresult. Both retain complete acceptance gates.
Two jobs currently running;no newparametergrid,finalevaluation orpromotion.


## Ranking-information follow-ups completed — 2026-10-18

Personalized recency cycle13 and annual-order exposure cycle14 completed after
separately frozen contingencies. Same period/class weights and HGB10parameters,
onlyone featureadded;no Octoberinputs. Recency recentAPgain+.03272,AUC+.03202,
AP95%interval[.00918,.05589],all3repeatpointgainspositive. TP changes+12,+7,+4;
FPchanges-4,+4,-12. MeanTPgain7.67,FP reduction4(1.84%). FailsrepeatFP,5%practical,
period/intersection and operatinguncertaintygates. Strongestprovisionalranking
information inthiscycle,butnotacceptedreplacement. Annualordercontrol APgain
+.01773,CI[-.00203,.03783],TPgain1,FPaddition3;failfullgates. Densityminusorder AP
increment+.00344,paired95%[-.02525,.03415],notuniquebreadthevidence. Allfailures
preserved,noextra-drawrescue. Fourarms1200savedHGBfits independentlyreloaded;
120cutoffs,2160operatingrows,5000operatingdrawsperarm checked,17rankingdrawspots.
Reports/checkpoint/sourceidentities:results/development-ranking-information-20261018.
No active fit or finalevaluation,promotion,commit,push or merge. Nextifspecifically
resumed:diagnosecutofftransfer919savedscorebeforeanynewtrainingorpolicy.

## Authorized personalized-recency October evaluation — 2026-10-18

User explicitly requested adding the new period/class + personalized recency
method to the existing20–70% precision-target October table. Method unchanged:
original10 + overdue_ratio, original HGB/seeds and fitting-only period/class
weights. Three grouped OOF folds seed78 on all1375/212/422development cases;
maximize recall subject to each precision floor, ties precision then cutoff.
20new retained HGB fits (15OOF,5full); no completed experiment rerun.
Predictions and cutoffs checkpointed before supplied October outcomes.
All six targets feasible. October precision/recall:20=11.41/100,
30=19.91/100,40=38.74/93.48,50=50.77/71.74,60=77.14/58.70,
70=86.36/41.30. At40:43TP68FP3FN378TN; compared with existing
period/class40:one additional caught churn case,17additional false alerts.
Development acceptance remains FAILED. October reused/descriptive, no new
independence or deployment claim. New table and complete counts/activity groups:
results/personalized-recency-october-20261018/evaluation/report.md.
The evaluation assigns a unique strategy display key per precision target for
the supplied-outcome join; frozen model scores/cutoffs/decisions unchanged.

### Consolidated lessons and documentation reconciliation — 2026-10-18

The latest authorized October result does not overturn the provisional ranking
finding or the failed development decision. At the 40% target, adding personalized
recency to period/class weighting catches one extra churner at the expense of
17 extra FP (43/68 versus 42/51). At 60%, it catches 27 versus 21 with eight
versus six FP. Thus 'worse at 40% for fewer false alerts' is supported; 'worse at
every operating point' is not. No business-approved minimum precision exists.

The missed-case audit establishes overlapping warnings, not absence of warnings.
The latest controlled arms test breadth, density, personalized timing and annual
order exposure; all full acceptance decisions fail. Density/count attribution
remains uncertain, and the reason for score/cutoff transfer after refitting is
not proven. Cohort composition partly explains development/October differences;
it does not establish a universal easy/hard customer rule.

Updated the existing model summary, feature evidence, evaluation metrics, rules,
README and current checkpoint. The complete 14-row 20%–70% table is now embedded
in evaluation_metrics.md rather than only in ignored generated results. That
section also records all new target confusion counts, latest four-arm validation
means, every personalized-recency repeat and evidence links. Historical journals,
source snapshots, acceptance decisions and result manifests remain intact.
This reconciliation uses saved artifacts only; no fitting or new evaluation.

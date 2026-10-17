# Persistent error investigation — 2026-10-16

## Cycle 1 protocol, frozen before matched-history inspection

Observed problem: the saved high-recall baseline persistently alerts on 353
recently purchasing non-churn cases and misses 11 recently purchasing churn
cases. All 63 continuing buyers without recent purchases are alerted.
Hypothesis: temporal order composition and purchasing changes omitted by the
ten inputs may distinguish errors from comparable correctly classified cases.
This is exploratory hypothesis generation, not causal inference, automatic
feature selection, or a performance comparison. No model or cutoff is selected.

Use the verified 1,375 development cases and saved three-repeat baseline
predictions only. Reproduce the 108 metric rows and 15-fold provenance audit
without fitting. Reconcile historical orders and lines to the preserved source
manifest and accepted order population. No October outcome files are inputs.
Order-time availability and source completeness remain assumptions; current
profiles, retrospective statuses, shipping outcomes and post-snapshot facts are
excluded. Main histories remain [t-12 months, t).

Compare three sets under the saved inner-selected policy:

1. Persistent recent-purchase false alerts versus recent-purchase churners
   caught in all three repetitions.
2. Churners missed in all repetitions versus recent-purchase non-churners
   correctly unalerted in all repetitions.
3. No-recent-purchase continuing buyers versus no-recent-purchase churners
   caught in all repetitions.

Each target is matched with replacement to the nearest three controls from the
same snapshot and activity group, excluding its reseller identity. Sensitivities
use one and five controls. Keep every target and report unavailable controls.
Similarity uses only the original ten features, never scores/outcomes/new
history descriptors. Transform order count and revenue with log1p; scale each
coordinate by its snapshot-wide development 5th–95th percentile range, clipping
observed differences at one. If this range is zero, use the full observed range,
then one if constant. Average ten absolute scaled differences; missing/observed
pairs cost one and missing/missing cost zero. No numeric imputation. These are
outcome-blind descriptive scales, not training preprocessing. Report each
coordinate contribution, distances, control reuse and ties at the kth boundary;
StoreID breaks distance ties reproducibly. There is no caliper or discarded
hard case. Scales are saved. Matching on existing features can suppress their
differences and is not evidence that those features lack predictive value.

Compare target minus mean-control descriptors per target, then summarize median,
mean, direction fractions and missingness, overall and per snapshot, without
p-values. Report raw groups as well. Reused controls, reseller snapshots,
selection on outcomes/errors and k sensitivities are dependent; matched counts
are not independent evidence. No confirmatory uncertainty claim is made.

Descriptors: annual count/span, cadence (mean gap, gap CV, overdue ratio, last
gap versus earlier mean gap), recent/previous-quarter order and revenue ratios,
last-order value versus earlier mean, basket quantity/lines/unit revenue,
last-order versus earlier category-share movement (total variation), last-order
product-set Jaccard overlap with earlier orders, annual unique products and
last-order product count. Undefined quantities stay missing. Last order is
selected by date then order ID; simultaneous orders can make that arbitrary,
so count and report same-day last orders. Save pre-snapshot order trajectories
and product sets for review. A no-previous-order case has no composition change.

Review previous spending/cadence/basket/diversity experiments before proposing
an additional model. A model cycle requires a separate frozen matched protocol
with baseline-recall preservation, practical FP reduction, paired reseller
uncertainty and subgroup constraints. No practical business margin was supplied.

## Cycle 1 findings

Completed under `results/persistent-error-investigation-20261016/`. Source
reconciliation retained 1,886 accepted pre-April orders and 28,356 lines; all ten
baseline features and six saved cadence/spending descriptors reconciled. The
current combined suite passed 210 tests, then four new synthetic history/matching
tests passed. Python is 3.14.8 (the available local interpreter), with the nine
principal packages pinned to the handoff versions; this is not the exact former
Python 3.12.14 runtime. No historical estimator bundle was deserialized.

All 353 recent false alerts, 11 missed churners and 63 inactive continuing buyers
have at least five same-snapshot controls. For three-neighbor matching, median
pair distance is 0.0824, 0.0945 and 0.0593 respectively; maxima are 0.5313,
0.2268 and 0.3349. Approximate neighbors can be poor matches. Recent false-alert
comparisons reuse only 91 control cases, up to 62 times; counts are dependent.

- Missed churners have fewer last-order lines than their three-control mean
  in 9/11 cases; median difference -3.33 lines. Median differences are -2.0,
  -3.33 and -3.6 with one/three/five controls. This direction holds in each
  represented snapshot (2 July, 2 January, 7 April cases); no persistent misses
  occur in October 2012. Annual variety is also lower, but strongly overlaps
  basket breadth. All 11 have estimable mean gaps; nine have gap CV. Sparse
  history is not a sufficient explanation for these misses.
- Recent false alerts have one extra annual distinct product at the median,
  compared with caught churners; the direction is positive in all four snapshots.
  Last-order breadth differences are weaker (median zero overall; +1 in July
  and October, zero in January/April). Last-order quantity/price differences are
  sensitive to neighbors and snapshot; no universal price rule is established.
- Category movement and revenue ratios reverse direction across snapshots.
  Last-gap-ratio and gap-CV medians largely overlap. Prior spending and cadence
  trials already examined related information, so these are not new remedies.
- Inactive continuing buyers and churners have heavily overlapping cadence and
  basket descriptors. Seventeen continuing buyers lack a prior annual order;
  41 lack gap CV. No clear signal explaining all 63 false alerts was found.
- Every historical case has one last-day order, so the date/order-ID choice is
  unambiguous here. Product-count and line-count are identical for these last
  orders; adding both would duplicate information.

These are outcome/error-conditioned comparisons from previously explored data,
not proof of predictive utility. Full signed differences, missingness, scales,
raw groups and trajectories are saved. A control mean uses available values;
`control_observed` and `partial_controls` expose incomplete comparisons.

Previous work: basket/diversity groups failed a seven-feature linear model's
old temporal operating objective; this does not test one basket descriptor added
to ten-feature boosting under matched nested grouped validation. Spending/cadence
and regularization already had repeated grouped trials and ablations. Reading
saved TP/FP tables shows many apparent precision gains fail baseline recall in
repeat 88. No historical method is relabelled as a newly validated winner.

Next bounded hypothesis: adding only last-order line count can expose basket
breadth not captured by annual category shares and money totals, helping ranking
near high recall. Test this minimal addition with baseline parameters unchanged.
Do not add both product count and lines, choose a price rule, or repeat the
already-completed basket/diversity bundle search.

## Additional historical-assignment screen (frozen before label comparisons)

While cycle 3 runs, inspect another available information family requested by
the continuation: order-header salesperson/territory histories. The clean-order
source retains these historical order fields, unlike mutable current Store
assignments. Source reconciliation historically verified them, but order-time
recording remains assumed. No current profiles or status fields are used.

At each of the four development snapshots, select accepted orders in the same
12-month window. Describe last salesperson and territory as categorical IDs
(never numeric predictors), number of distinct salespeople, and whether the
observed window contains more than one salesperson. "Change" is an observed
within-window assignment difference, not a verified relationship event. Join the
same fixed matched pairs from cycle 1; report target/control switch fractions
and categorical shares overall and by snapshot, plus full-development categorical
churn counts/rates with small-cell flags. Controls can be reused. No p-values,
causal claims, automatic feature selection, candidate fitting or cutoff changes.
This screen addresses a historical information family not covered by basket or
cadence trials. Historical raw source fields receive no stronger availability
claim from this descriptive screen.

## Archived-value discrepancy screen (frozen before case comparisons)

The live CompanyX audit found archived transaction ActualCost values different
from current order UnitPrice, while archive quantities/dates reconcile. Catalog
semantics and the current trigger do not resolve this discrepancy. Use the same
fixed cycle-1 matched pairs to investigate whether it warrants source-owner
clarification. This is a numeric discrepancy screen, not a margin/price-revision
predictor, causal claim or automatic feature selection.

For accepted archived sales lines, compute sum(quantity*ActualCost) divided by
sum(quantity*current UnitPrice), separately for the last order and all orders in
[t-12 months,t). All transaction dates must also precede t. Values in the two
sources may represent different concepts/versions; interpreting the ratio as
profit, discount, or known historical pricing is explicitly prohibited without
clarification. Summarize target-minus-mean-control differences by the same
one/three/five neighbors and snapshots, preserving dependence/missingness.
If patterns are unstable, do not infer absence of all useful product-level
information. If patterns are stable, resolve meanings and historical availability
before ordinary model fitting; no October outcomes or new cutoffs are involved.

## Product identity mix screen — frozen before SKU outcome summaries

The ten-feature baseline represents broad categories, while prior extra-feature
trials tested diversity counts and own-history product overlap. Neither represents
which historical SKUs were bought. After fixed learner/weight contrasts fail,
this remains a distinct available historical information family. ProductID on an
accepted order line is an order fact, not a mutable reseller profile. Use those
recorded event IDs and accepted line revenue only; no current product attributes,
end-of-sale dates, statuses or unresolved archive ActualCost fields.

Build each case's annual and latest-order revenue shares over all historical
ProductIDs using only accepted purchases in [t-12 months,t). Latest order is
ordered by date then order ID (one last-day order in these data). Normalize by
sum of accepted line revenue; this makes the SKU shares sum to one. Reconcile
line revenue with accepted order subtotals within the established rounding
contract. All source hashes and per-snapshot boundaries must be checked.

Use the fixed cycle-1 one/three/five-control pairs. Report every SKU's target
minus mean-control revenue share, raw means/presence, overall and by snapshot;
report pairwise total-variation distance (half sum absolute share differences)
for annual/latest SKU profiles. Controls and reseller snapshots remain dependent.
No p-values, causal claims or automatic feature selection. A one-percentage-point
share difference in the same direction across all four snapshots is only a
predeclared descriptive review flag; retain unflagged/negative findings too.
Small missed-churn groups remain small, regardless of the number of SKU cells.

This screen can motivate a richer representation, not a selected list of SKUs.
Any subsequent predictive vocabulary must be learned from training cases only,
without using these outcome contrasts to choose specific product IDs. It would
need a separate frozen protocol with the unchanged operational acceptance rule.

## Observed pre-annual history — frozen before extended-history contrasts

2026-10-17. Product mix preserves original repeat recall but fails the complete
FP/practical/uncertainty objective; no vocabulary tuning follows. Cadence-only
and SKU models continue to alert nearly every inactive continuing buyer. Check
whether truncating history at12 months discards earlier observed cadence that
could distinguish irregular continuing buyers. An earlier alternative method
used all observed history, but changed populations, features, pruning and learners;
its documentation does not isolate this extension under the current objective.
Those older bundles are prior evidence, not proof this question is untested or
that longer history must help.

Use only accepted orders before each approved snapshot; source coverage starts
2011-05-31. Its completeness is assumed, not independently verified. A zero
older-order count means no earlier OBSERVED order in that export, not a new
reseller. Exposure before the main annual window ranges with snapshot, so fixed
same-snapshot controls are essential; no snapshot is a predictor. Do not extend
labels, change eligibility, use future orders or reinterpret first purchase as
true opening. This is an exploratory no-fit screen, not model selection.

Describe total observed orders/span, count strictly before t-12m, mean and CV of
all distinct-date interpurchase gaps, all-history recency/mean-gap overdue ratio,
and last gap/mean earlier-gap ratio (the last requires two gaps). Compare the
all-history overdue ratio with the annual version and report how often previously
undefined annual cadence becomes observable. Preserve missing ratios; no future
fills. Retain all fixed1/3/5 matched pairs, raw target/control values and missing
counts, overall/by snapshot, using the existing paired-summary utilities.

The specific hypothesis predicts LOWER full-history overdue ratios for recent
false alerts and inactive continuing buyers than their churn controls, HIGHER
for missed churners than correct continuing controls. Report every direction and
neighbor sensitivity. Reversals, large missingness, tiny missed-churn groups and
left truncation all limit interpretation. Neither a favorable mean nor sparse
history automatically authorizes an added feature; any fit needs a separately
frozen matched protocol and unchanged acceptance requirements.

## Exact observable-history collision audit — frozen before collision counts

2026-10-17. Extended overdue contrasts reverse; no added cadence feature is
selected. Product ranking improves descriptively but training-to-outer operating
metrics do not establish a uniform cutoff-transfer failure. Before blaming
irreducible overlap, check whether opposing outcomes actually share identical
available information. This is no-fit descriptive outcome-conditioned research.

Compare exact10-feature vectors (explicit missing tokens) and exact accepted
purchase histories, both annual and all observed pre-snapshot. A history is a
sorted multiset of orders described by age in days at prediction, SubTotal, and
sorted line tuples ProductID,OrderQty,UnitPrice,UnitPriceDiscount,LineTotal. Ignore
StoreID/order/detail IDs as predictors; retain them only for joining/audit. No
rounding/tolerance, current attributes, archive cost, future orders or labels in
signatures. Different histories must not be called indistinguishable merely
because a chosen summary collapses them. Same-snapshot matching prevents a
collision assertion from assuming calendar/snapshot interchangeability.

For every representation report distinct signatures, duplicate rows/resellers,
and mixed-outcome cells, by snapshot and overall (overall still includes snapshot
in the grouping). Join original persistent-error groups and retain each case's
cell size/churn counts. Also report collisions within each saved outer fold and
baseline decisions. No oracle cutoff/model selection or causal/Bayes-error claim.
Mixed cells demonstrate limitations of that representation in this finite sample;
absence of exact collisions is NOT evidence that learnable separation exists.
Source completeness/order-time availability assumptions remain unresolved.

## Frozen-vocabulary coverage audit — before coverage/outcome summaries

2026-10-17. Both fixed SKU and bootstrap variants fail operational acceptance;
no vocabulary cap/support search follows. Audit what the original fixed20-product
representation actually captured. For each unbagged SKU outer fold use its saved
TRAINING-selected vocabulary. For every held-out case compute annual/latest
revenue fraction assigned to explicit SKUs versus other. Report all cases,
snapshots, activity, and TP/FP/FN/TN of the already saved decisions, with sample
sizes and median/mean coverage and zero-coverage fractions. Preserve individual
case coverage. These groups are outcome-conditioned and dependent; low coverage
is not demonstrated error cause. No unseen/rare ID becomes a selected feature,
no new cutoff or fit, no bag-count/vocabulary tuning or protected outcomes.

## Cadence residual basket screen — frozen before residual-group summaries

2026-10-17. Crossfit percentile reduces FP but loses a baseline-caught churner;
raw crossfit misses practical/uncertainty rules. No rule/margin/reference tuning
follows. Return to interpretable historical information: the fixed cadence12
procedure has different errors from original HGB10. Original baseline matching
found recent persistent false alerts had lower latest-order revenue/quantity
(mean-108.94,median-92.19) than caught churners; inactive continuing buyers also
lower(mean-48.85,median-54.58). This paid unit revenue is NOT archived cost/profit.
It is an accepted subtotal divided by accepted summed quantity. Latest-line count
and annual diversity already failed on original31-leaf HGB; do not hide those
negative findings or automatically repeat them. No prior isolated trial used
last-order unit revenue, based on executable/doc search.

Use saved cadence-no-count predictions on301/919/1201 to define alerts across3
repetitions and the SAME three error/control roles, same-snapshot ten-feature
missingness/scaled matching and k1/3/5 as original diagnosis. Reuse all17 verified
history descriptors and show all missingness/directions, not only a favorite.
Primary exploratory hypothesis: lower latest paid unit revenue continues to
distinguish recent persistent false alerts and inactive continuing buyers from
caught churn controls after cadence modeling. The missed group may be smaller;
report actual counts, no assumed sparse-history cause or significance claims.
Compare signs/neighbor sensitivity across snapshots; no automatic feature
selection, no fits/cutoffs, and no protected outcomes. A new feature trial needs
its own frozen matched protocol and unchanged baseline-recall/FP acceptance.

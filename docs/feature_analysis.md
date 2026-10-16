# Feature definitions and development evidence

Reviewed on 2026-10-10 against current code and existing local result artifacts.
No models were retrained for this documentation. Definitions come from executable
code; measured usefulness comes from completed development reports. Possible
explanations are identified as hypotheses rather than established causes.

The authorized 2026-10-11 final evaluation now uses the same ten-feature
definitions with historical October inputs and saves a separate provisional
ensemble. It does not register a new main-pipeline feature set. October outcomes
are now inspected; the earlier development comparisons below retain their
original evidence. See [final results](evaluation_metrics.md#final-test-evaluation--2026-10-11).

## How features enter the current workflow

Preparation builds 17 predictors. Model development compares four subsets of
those predictors across Logistic Regression, Random Forest, and
HistGradientBoosting. The chosen subset can differ for each precision minimum;
there is no single feature set selected for every model or operating rule.
Separately, root `debug/` constructs CompanyX order-line features and compares
a seven-feature baseline with ten features. These experimental additions are
not registered in the main pipeline's feature-set configuration.

The sets are defined in
[feature_sets.py](../reseller_churn/config/feature_sets.py):

| Set | Count | Included features |
| --- | ---: | --- |
| RFM (`rfm`) | 3 | `recency_days`, `n_orders_12m`, `revenue_12m` |
| Pruned (`pruned`) | 7 | `n_orders_3m`, `revenue_3m`, `recency_days`, `share_bikes`, `share_accessories`, `share_clothing`, `revenue_12m` |
| Full (`full`) | 17 | All predictors in the catalog below |
| Reduced (`reduced`) | 14 | Full excluding `revenue_6m`, `overdue_ratio`, and `share_components` |

To find the inputs of a saved predictor, open its selection manifest under
`results/<run_id>/development/selection/` and read `candidate.feature_set` and
`candidate.features`. A manifest with `status: no_feasible_candidate` has no
selected model. See [pipeline execution](../pipeline/README.md) for commands
and [evaluation metrics](evaluation_metrics.md#current-development-selection-reports)
for the recall and precision selection rule.

`StoreID`, `snapshot`, and `churn` identify rows or provide the target; they are
excluded from model inputs. The seven-feature Pruned set was used by the
historical provisional scripts. Its measured evidence is recorded below.

## Main feature catalog

At snapshot `t`, all order-based inputs are restricted to `[t - 12 months, t)`.
Recent periods are `[t - m months, t)` for `m` equal to 3 or 6. Revenue means
order `SubTotal`, excluding tax and freight. Sources refer to
[the clean dataset mappings](data_analysis.md#clean-order-dataset).

| Feature | Definition and source | Potential modeling role |
| --- | --- | --- |
| `recency_days` | Days from the latest historical `OrderDate` to `t` | Measures how long the reseller has been inactive |
| `n_orders_3m` | Distinct `SalesOrderID` count in the preceding 3 months | Recent purchase frequency |
| `revenue_3m` | Sum of `SubTotal` in the preceding 3 months | Recent purchase volume |
| `revenue_12m` | Sum of `SubTotal` over the 12-month history | Longer-term scale of the relationship |
| `share_bikes` | Historical `rev_bikes` divided by `revenue_12m` | Product mix and possible purchase-cycle differences |
| `share_accessories` | Historical `rev_accessories` divided by `revenue_12m` | Product mix |
| `share_clothing` | Historical `rev_clothing` divided by `revenue_12m` | Product mix |
| `n_orders_12m` | Distinct order count over the 12-month history | Longer-term purchase frequency |
| `n_orders_6m` | Distinct order count in the preceding 6 months | Medium-term purchase frequency |
| `revenue_6m` | Sum of subtotal in the preceding 6 months | Medium-term purchase volume |
| `mean_gap` | Mean gap in days between distinct historical order dates | Reseller's usual purchase cadence |
| `std_gap` | Sample standard deviation of those gaps | Irregularity of purchase cadence |
| `overdue_ratio` | `recency_days / mean_gap` | Inactivity relative to usual cadence |
| `has_previous_6m_revenue` | 1 if `revenue_12m - revenue_6m > 0.01`, else 0 | Distinguishes resellers with earlier activity |
| `revenue_trend` | `log1p(revenue_6m) - log1p(revenue_12m - revenue_6m)` when earlier revenue exceeds 0.01; otherwise missing | Recent growth or decline relative to the earlier half-year |
| `share_components` | Historical `rev_components` divided by `revenue_12m` | Product mix; overlaps with the other three shares |
| `store_age` | `t.year - stores.YearOpened` | Store maturity measured in calendar years |

Definitions are implemented in
[feature_pipeline.py](../reseller_churn/features/feature_pipeline.py).
Category revenue originates in order lines joined to product categories. Gap
calculations deduplicate order dates: multiple orders on the same day do not
create zero-day gaps. Store age uses the survey opening year, not the first order
date. Its source evidence and historical-availability assumption are owned by
[data_analysis.md](data_analysis.md#opening-year-assumption-and-provenance-limits).

## Experimental ten-feature set

The completed debug comparison uses the seven Pruned inputs plus three discount
inputs. It is an experimental set, not a replacement for the registered four
sets. At prediction date `t`, every feature uses orders strictly before `t`.

| Input | Plain meaning | Window |
| --- | --- | --- |
| `n_orders_3m` | Number of distinct recent orders | Latest 3 months |
| `revenue_3m` | Recent subtotal spending, excluding tax and freight | Latest 3 months |
| `recency_days` | Days since the most recent historical order | Latest order in 12-month history |
| `share_bikes` | Fraction of annual spending on bikes | Latest 12 months |
| `share_accessories` | Fraction of annual spending on accessories | Latest 12 months |
| `share_clothing` | Fraction of annual spending on clothing | Latest 12 months |
| `revenue_12m` | Annual subtotal spending | Latest 12 months |
| `weighted_discount_3m` | Average discount weighted by quantity times undiscounted unit price | Latest 3 months |
| `discount_change_3m` | Recent weighted discount minus the previous quarter's weighted discount | Latest 3 months versus months 3–6 before prediction |
| `discounted_line_share_12m` | Fraction of order lines receiving a positive discount | Latest 12 months |

For a quarter, weighted discount equals
`sum(OrderQty * UnitPrice * UnitPriceDiscount) / sum(OrderQty * UnitPrice)`.
Larger undiscounted purchases receive more weight. Discount change is a
difference between fractional rates: 0.05 means an increase of five percentage
points, not five percent relative growth. Discounted-line share counts lines,
not orders or purchased units; a 1% and a 30% discount both count as discounted.

Recent weighted discount is missing when there are no recent purchases.
Discount change requires purchases in both quarters. Zero discount means
purchases occurred without a discount and is distinct from missing activity.
The annual discounted-line share is observed for all 1,375 eligible development
rows; recent weighted discount is observed for 1,221 and discount change for
942. LR/forest use training-fitted imputation with indicators; boosting handles
missing values natively.

The builder is [build_companyx_features.py](../debug/build_companyx_features.py);
the model comparison is [compare_seven_ten_features.py](../debug/compare_seven_ten_features.py).
Inputs originate in the historical CompanyX headers and lines described in
[data analysis](data_analysis.md#historical-order-line-extract-for-debugging).
Order-time availability is assumed, not verified through recording-time history.

### Measured usefulness of the new features

The initial fixed Logistic Regression experiment evaluated three groups
separately: basket volume (five inputs), discounts (three), and product diversity
(three). None improved mean temporal recall over the seven-feature baseline;
discounts preserved recall with three fewer false alerts. Correlation and
training redundancy screening were descriptive, not feature selection.

The subsequent three-architecture comparison found that the discount group
improves boosting recall under temporal and grouped validation, preserves
Logistic Regression temporal recall while reducing false alerts, and lowers
Random Forest ensemble recall under both methods. Individual forest-seed
benefits vary. The discount bundle's value is therefore architecture-dependent;
no isolated per-discount-feature contribution has been established.

See [latest metrics](evaluation_metrics.md#latest-completed-experiment-seven-versus-ten-features)
for scores and costs, and [debugging.md](debugging.md) for all 11 candidate
definitions, associations, paired prediction changes, and experiment limits.

## Historical feature-set comparison

Existing Stage 07 output reports the following mean development PR-AUC, using
the project's average-precision convention:

| Feature set | Count | Random Forest | HistGradientBoosting |
| --- | ---: | ---: | ---: |
| Pruned | 7 | 0.3374 | 0.2915 |
| Reduced | 14 | 0.2683 | 0.2774 |
| Full | 17 | 0.2515 | 0.2337 |
| RFM | 3 | 0.2121 | 0.2231 |

Evidence: [summary](../results/feature_selection/01_feature_set_summary.csv) and
[per-fold results](../results/feature_selection/01_feature_set_folds.csv),
generated by [Stage 07](../pipeline/07_feature_selection/01_compare_feature_sets.py).
The comparison uses the two standard temporal folds, baseline 300-tree Random
Forest averaged over seeds 0–19, and baseline HistGradientBoosting at seed 42.
These scores describe the historical feature-set protocol. They do not measure
the new recall-based search or the tuned historical provisional models.
Generated result files are ignored by Git and may be absent in a fresh
checkout; the values above preserve the inspected evidence.

The seven-feature set scored higher than the alternatives on each of the two
folds for both architectures under that average-precision protocol. There is
no recorded isolated ablation establishing the gain from each feature.
These results do not establish which set wins under the current recall and
precision rule.

### A result that limits blanket rejection

The separate controlled strategy experiment searched the same four named sets
but omitted `store_age`, giving 3, 7, 13, and 16 predictors. GroupKFold selected
the 16-feature Full Random Forest without resampling; temporal selection chose
the 3-feature RFM Random Forest with oversampling. On the same April 2013
population, their PR-AUC values were 0.4183 and 0.2813, respectively. The formal
winner rule returned **inconclusive**.

Evidence: [selection manifest](../results/validation_strategy_experiment/frozen_selection_manifest.json),
[common metrics](../results/validation_strategy_experiment/common_evaluation_metrics.csv),
and [experiment summary](../results/validation_strategy_experiment/experiment_summary.json).
This retrospective experiment changes selection design and chosen imbalance
treatment as well as feature count, uses partial April 2012 history, and reuses
a previously inspected development period. It does not isolate the effect of
individual features. The omission of `store_age` preserves its registered search
and is not evidence of leakage or predictive uselessness.

## Historical feature diagnostics

The following diagnostics use all 1,375 main labeled development rows. They are
descriptive Pearson correlations, not fold-local feature-selection evidence or
causal explanations. Missingness is measured on the same rows.

| Feature | Missing | Correlation with churn | Interpretation and limits |
| --- | ---: | ---: | --- |
| `n_orders_3m` | 0% | -0.4295 | Strongest absolute linear association in this report |
| `recency_days` | 0% | +0.3414 | Longer inactivity associates with churn |
| `revenue_3m` | 0% | -0.2198 | Recent spending associates with retention |
| `revenue_12m` | 0% | -0.0976 | Weak marginal association; no isolated gain measured |
| `share_bikes` | 0% | +0.0108 | Near-zero marginal linear association; interactions may still matter |
| `share_accessories` | 0% | -0.0470 | Weak marginal association; no isolated gain measured |
| `share_clothing` | 0% | -0.0156 | Weak marginal association; no isolated gain measured |
| `n_orders_12m` | 0% | +0.0276 | Little marginal signal; overlap with other frequency windows is a hypothesis |
| `n_orders_6m` | 0% | -0.1667 | Has marginal signal; overlap with recent frequency may limit incremental benefit |
| `revenue_6m` | 0% | -0.1600 | Has marginal signal; overlapping revenue windows may be redundant |
| `mean_gap` | 18.25% | +0.1167 | Undefined without two distinct order dates; sparse history may limit usefulness |
| `std_gap` | 37.82% | +0.0376 | Needs at least two gaps for sample standard deviation; sparse history and weak marginal signal |
| `overdue_ratio` | 18.25% | +0.3394 | Strong marginal association, but inherits missing cadence; exclusion is not proof of no signal |
| `has_previous_6m_revenue` | 0% | +0.1310 | Describes history coverage/activity; no isolated incremental benefit measured |
| `revenue_trend` | 32.80% | -0.0935 | Undefined without earlier revenue; missingness and sensitivity to history are plausible limitations |
| `share_components` | 0% | +0.0003 | Almost no marginal linear association; the four shares sum approximately to one, creating overlap |
| `store_age` | 0% | +0.0059 | Very weak marginal linear association; no isolated benefit measured, and no leakage demonstrated |

Evidence: [feature-quality report](../results/feature_analysis/01_feature_quality_summary.csv),
generated by [Stage 05.1](../pipeline/05_feature_analysis/01_analyze_feature_quality.py).
The historical Full/Reduced-versus-Pruned evidence is at the bundle level. Sparse
history, overlapping inputs, and added model complexity are plausible explanations
for weaker temporal performance, but no controlled per-feature ablation has
confirmed these causes. There is currently no supported list of individual
features proven never to work.

## Additional features tried in the alternative method

The [alternative builder](../comparisons/correlation_pruned_churn/corrected_features.py)
constructs the following additional inputs. They are outside the four feature
sets in the supported development search. Its `full` variant uses all past
observed orders; `obs6` uses only the preceding six months. Identically named features can therefore have different
history windows from the main pipeline.

| Feature | Extraction and intended use |
| --- | --- |
| `tenure_days` | Days since the first order in the variant's observed history; relationship length, not true store age |
| `n_orders_total` | Distinct order count across the variant's history; relationship frequency |
| `revenue_total` | Total subtotal across that history; relationship value |
| `aov` | Mean order subtotal; typical basket value |
| `avg_qty` | Mean per-order summed quantity; typical basket quantity |
| `avg_lines` | Mean per-order detail-line count; basket breadth |
| `avg_disc` | Mean of per-order mean line discounts; discount behavior, not a revenue-weighted discount |
| `rev_trend` | Recent-half revenue divided by previous-half revenue when positive; a ratio rather than the main log difference |
| `TerritoryID` | Territory from the last past order; historical categorical context |
| `last_order_salesperson` | Salesperson from the last past order; historical categorical context |

The ratio compares 6 versus previous 6 months in `full`, and 3 versus previous
3 months in `obs6`. Its configured numeric candidate list includes `n_orders_3m`,
but this builder constructs 6/12-month counts in `full` and 3/6-month counts in
`obs6`; only columns present in the built variant enter fitting.

The optional store-profile diagnostic additionally uses `AnnualSales`,
`AnnualRevenue`, `SquareFeet`, `NumberEmployees`, `BusinessType`, `Specialty`,
`Brands`, `Internet`, and `store_salesperson`, from the static store export.
Their historical availability is unverified. Diagnostic exclusion is a provenance
decision rather than a measured claim that they lack predictive value. See
[feature_provenance.py](../comparisons/correlation_pruned_churn/feature_provenance.py).

The Stage 06 factor-audit artifact includes a narrower nine-feature overlap:
`n_orders_6m`, `n_orders_12m`, `mean_gap`, `std_gap`, `overdue_ratio`,
`share_bikes`, `share_clothing`, `share_accessories`, and `store_age`. With the
same alternative 500-tree Random Forest settings and seed 42, temporal mean
PR-AUC was 0.3283 for Pruned, 0.2782 for Full, and 0.2260 for this overlap set.
Under that audit's GroupKFold design, the corresponding values were 0.5765,
0.5957, and 0.5963. This again shows that a feature bundle's ranking can depend on
the evaluation question. The grouped and temporal scores are not matched-target
estimates of the same quantity.

Evidence: [factor-audit report](../results/model_selection/02_friend_approach_factor_comparison.csv)
and [its executable protocol](../pipeline/06_model_selection/02_compare_model_design_factors.py).
Local artifacts for the alternative builder's individual feature contributions
were not available in this inspection. Do not infer which additional feature
failed from its absence in the current candidate sets.

## Missing values and next research decisions

Random Forest and Logistic Regression use training-fitted median imputation
with missing indicators; Logistic Regression also scales inside its pipeline.
HistGradientBoosting handles missing values natively. No future orders should
fill missing historical behavior. Changing feature definitions, adding a
feature, or declaring one unnecessary requires a documented development
comparison with matched evaluation targets, fixed selection rules, and reported
recall, precision, false alerts, and ranking metrics. The protected final test
must not guide those choices.

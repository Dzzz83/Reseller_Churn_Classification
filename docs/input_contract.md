# Input, time, and feature contract

Every source CSV has a header row. Order dates, scoring snapshots and declared
coverage endpoints must be timezone-naive calendar dates at midnight. Intraday
orders are rejected to preserve the trained calendar-day recency formula.

| File | Required columns |
| --- | --- |
| Clean orders | StoreID, SalesOrderID, OrderDate, SubTotal, qty, n_lines, rev_bikes, rev_components, rev_clothing, rev_accessories |
| Stores | StoreID |
| Headers | StoreID, SalesOrderID, OrderDate, SubTotal |
| Lines | SalesOrderID, SalesOrderDetailID, ProductID, OrderQty, UnitPrice, UnitPriceDiscount, LineTotal |

Extra source columns are allowed. Store/order/detail keys must be present and
unique at their respective grains. Source lines must reference a header.
Every accepted historical order must have a matching header and lines; reseller
mapping, date, quantity, line count, and revenue must reconcile. Prices,
quantities, and accepted revenue are positive; discounts are fractions in
[0, 1]. Category revenue sums match subtotals within 0.01, detail revenue within
0.02, and line formulas within rounding tolerances.

Historical fixed training keeps its original CSV parsing protocol. Supported
grouped development, scoring features and prediction CSVs use round-trip float
parsing: decimal serialization must
not move a feature across a fitted histogram boundary or a score across a
user-supplied cutoff.

Accepted clean order IDs define the historical population. Any historical raw
header omitted from the clean orders must be listed explicitly in independently
approved excluded order IDs. Otherwise preparation fails rather than treating a
missing accepted order as an intentional rejection. Approved and actual excluded
IDs are recorded in manifests.
The pipeline does not reclassify orders from current cancellation status.
Source facts are assumed available when the order was placed; recording-time
versions remain unverified. The declared coverage interval must be complete,
including days with no transactions. This requires source knowledge beyond
what the pipeline can infer from a CSV's first/last purchase. `loaded_through`
records the exclusive cutoff actually retained in memory; feature and label
requests must fit both the declared interval and the loaded interval.

For snapshot t, eligible resellers purchased in `[t - 6 months, t)`. Features
use only transactions in `[t - 12 months, t)` and recent periods strictly before
t. The churn outcome is no accepted purchase in `[t, t + 6 months)`.
All windows include their start and exclude their end. Discount absence is
missing, not zero discount. Missing discount values use the estimator's native
handling; the seven base features must be finite.

| Feature | Formula |
| --- | --- |
| n_orders_3m | Unique orders in the preceding three months; zero if none |
| revenue_3m | Sum of accepted subtotals in the preceding three months; zero if none |
| recency_days | Calendar days since the most recent accepted order before t |
| share_bikes, share_accessories, share_clothing | Category revenue / total revenue in the preceding twelve months |
| revenue_12m | Sum of accepted subtotals in the preceding twelve months |
| weighted_discount_3m | Sum(gross line value × discount) / sum(gross line value), gross = quantity × unit price |
| discount_change_3m | Recent three-month weighted discount minus the preceding three-month weighted discount |
| discounted_line_share_12m | Fraction of line rows in the preceding twelve months with discount > 0 |

Imported ten-feature tables require unique positive reseller IDs, calendar
snapshots and finite base features. Recency must fit the six-month eligibility
window. Annual revenue must be positive, recent revenue cannot exceed it, recent
counts and revenue must agree, and the three included category shares cannot sum
to more than one (with a small floating-point tolerance). Missing discounts remain
native missing values.

`export` requires an unlabeled features CSV and its feature-stage manifest. The
file checksum, row/snapshot population, declared annual coverage and effective
loaded cutoff are checked. A valid manifest records source declarations; it
cannot independently prove source completeness or recording-time availability.

The export uses existing estimators only. It writes StoreID, snapshot,
n_orders_3m, churn_probability, threshold, predicted_churn, model_id and rank.
The score is the mean seeded probability, and predicted_churn is one when the
full-precision score is greater than or equal to the explicit threshold. The
export manifest records model and feature identities, the original training
source metadata, comparison rule and zero models fitted. Outcomes are never
required or read.

Evaluation accepts saved unlabeled predictions with StoreID, snapshot,
churn_probability, and n_orders_3m; rank is optional. The separate outcome CSV
contains StoreID, snapshot, actual_churn. Scores must be finite in [0, 1], counts
nonnegative integers, labels binary, and account/snapshot populations identical.
The outcomes-through declaration must reach each snapshot's six-month endpoint.
Existing saved outcomes may be used directly; no label generation is performed
by evaluation. Labels from external sources must follow the same purchase and
eligibility contract.

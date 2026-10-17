# Input, time, and feature contract

Every source CSV has a header row. Dates are timezone-naive; scoring snapshots
and declared coverage endpoints must be calendar dates at midnight.

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

Source and training CSV parsing preserve the original protocol. Scoring feature
and prediction CSVs use round-trip float parsing: decimal serialization must
not move a feature across a fitted histogram boundary or a score across a
user-supplied cutoff.

Accepted clean order IDs define the historical population. Raw orders omitted
by established cleaning are excluded, with their IDs recorded in manifests.
The pipeline does not reclassify orders from current cancellation status.
Source facts are assumed available when the order was placed; recording-time
versions remain unverified. The declared coverage interval must be complete,
including days with no transactions. This requires source knowledge beyond
what the pipeline can infer from a CSV's first/last purchase.

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
| discount_frequency_12m | Fraction of order lines in the preceding twelve months with discount > 0 |

`discount_frequency_12m` was previously named `discounted_line_share_12m`; its
formula and position in the feature schema are unchanged. Regenerate development
and feature tables for the updated pipeline. Preserve historical artifacts with
their original names and recorded code; existing model bundles still require
their recorded code version or explicit retraining.

Evaluation accepts saved unlabeled predictions with StoreID, snapshot,
churn_probability, and n_orders_3m; rank is optional. The separate outcome CSV
contains StoreID, snapshot, actual_churn. Scores must be finite in [0, 1], counts
nonnegative integers, labels binary, and account/snapshot populations identical.
The outcomes-through declaration must reach each snapshot's six-month endpoint.
Existing saved outcomes may be used directly; no label generation is performed
by evaluation. Labels from external sources must follow the same purchase and
eligibility contract.

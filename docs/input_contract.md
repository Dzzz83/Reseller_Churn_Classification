# Input, time, and feature contract

Every source CSV has one header row, without repeated column names. Dates are
present, timezone-naive calendar dates at midnight. Snapshot and coverage dates
use the same rule. CSV numeric parsing preserves round-trip binary64 precision.

| File | Required columns |
| --- | --- |
| Clean orders | StoreID, SalesOrderID, OrderDate, SubTotal, qty, n_lines, rev_bikes, rev_components, rev_clothing, rev_accessories |
| Stores | StoreID |
| Headers | StoreID, SalesOrderID, OrderDate, SubTotal |
| Lines | SalesOrderID, SalesOrderDetailID, ProductID, OrderQty, UnitPrice, UnitPriceDiscount, LineTotal |

Extra source attributes are allowed, but outcomes/predictions are rejected.
The scoring boundary accepts exactly StoreID, snapshot and the ten features.
Keys must be present and unique at their grain. IDs and counts/quantities are
positive integers. Accepted orders reference known stores; lines reference
headers. Every accepted historical order must have matching headers and lines.
Reseller mapping, date, quantity, line count and revenue reconcile. Prices and
accepted revenue are positive; discounts are fractions in [0, 1]. Category sums
match subtotals within 0.01, detail totals within 0.02 plus relative tolerance
1e-8; line formulas use absolute tolerance 1e-5 and relative tolerance 1e-8.

## Coverage and exclusions

Provide `--source-manifest` using this schema. Replace the illustrative hashes
with SHA256 values of the exact supplied CSV files, and have the source owner
attest completeness. Dates are examples, not established source readiness.

```json
{
  "schema": "source-coverage-v1",
  "attested_by": "source owner / extraction record identifier",
  "approved_excluded_order_ids": [],
  "sources": {
    "orders": {
      "sha256": "<orders CSV SHA256>",
      "coverage_from": "2013-05-02",
      "coverage_through": "2014-05-02"
    },
    "stores": {"sha256": "<stores CSV SHA256>"},
    "headers": {
      "sha256": "<headers CSV SHA256>",
      "coverage_from": "2013-05-02",
      "coverage_through": "2014-05-02"
    },
    "lines": {
      "sha256": "<lines CSV SHA256>",
      "coverage_from": "2013-05-02",
      "coverage_through": "2014-05-02"
    }
  }
}
```

`sha256sum inputs/*.csv` can calculate local hashes. Hashes bind declarations
to files; they do not establish completeness by themselves. The effective
coverage is the intersection of orders/header/line coverage. Both the requested
`--coverage-from`/`--coverage-through` and the required feature window must fit
inside it. The effective loaded endpoint is also bounded by the historical
load cutoff. The output manifest records these separately. Inputs are hashed
before and after loading; changing files are rejected.

Coverage endpoints are exclusive and include days with no transactions.
Maximum transaction dates cannot prove completeness or the absence of orders.
All historical accepted clean orders must be present in source headers/details;
missing recent headers cannot be compensated by a newer order-summary date.
The completeness attestation still requires independent extraction/source review.

Accepted clean order IDs define the population. Every historical header absent
from clean orders must appear in `approved_excluded_order_ids`; otherwise loading
fails. Actual exclusions are recorded separately from approvals. Approval must
come from independent cleaning/reconciliation evidence. The pipeline does not
reinterpret current cancellation status. Header/line rows for approved exclusions
are removed before accepted-value reconciliation. Duplicate historical raw keys
still fail. Source facts are assumed available when the order was placed;
recording-time versions remain unverified.

## Historical windows and features

For snapshot t, eligible resellers purchased in `[t - 6 months, t)`. Features
use only transactions in `[t - 12 months, t)`. Churn means no accepted purchase
in `[t, t + 6 months)`. All boundaries use calendar-month offsets, including
starts and excluding ends. Only approved development snapshots generate labels;
export never constructs outcomes. Future transaction rows are excluded from
feature calculation.

| Feature, in estimator order | Formula |
| --- | --- |
| n_orders_3m | Unique orders in preceding three months; zero if none |
| revenue_3m | Accepted subtotals in preceding three months; zero if none |
| recency_days | Calendar days since last accepted order before t |
| share_bikes | Bike revenue / total revenue in preceding twelve months |
| share_accessories | Accessory revenue / total revenue in preceding twelve months |
| share_clothing | Clothing revenue / total revenue in preceding twelve months |
| revenue_12m | Accepted subtotals in preceding twelve months |
| weighted_discount_3m | Sum(gross line value × discount) / sum(gross line value), gross = quantity × unit price |
| discount_change_3m | Recent three-month weighted discount minus preceding three-month weighted discount |
| discount_frequency_12m | Fraction of preceding twelve-month line rows with discount > 0 |

Discount absence is missing, not zero discount. The estimator handles missing
values natively; the seven base features must be finite. Eligibility constrains
recency; recent revenue cannot exceed annual revenue, category shares cannot
sum above one, and recent order counts and revenue must agree.

## Power BI CSV

The exact columns are:

```text
StoreID,Snapshot,n_orders_3m,revenue_3m,recency_days,share_bikes,share_accessories,share_clothing,revenue_12m,weighted_discount_3m,discount_change_3m,discount_frequency_12m,Model score,Predicted churn
```

There is one row per eligible reseller at one snapshot, in ascending StoreID
order. Snapshot is `YYYY-MM-DD`. Features are exactly the validated estimator
inputs; there is no imputation or rounding. Numeric decimals round-trip; missing
discounts are empty fields. Flags are `Yes` for score >= the manifest cutoff,
otherwise `No`. Threshold and provenance belong in the sidecar manifest.
The pipeline rejects empty eligible populations and existing output directories.

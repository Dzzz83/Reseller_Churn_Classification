# CompanyX database and dataset analysis

Database metadata and source-to-CSV comparisons were inspected on 2026-10-10.
This document records measured source facts and their limits; the prediction
contract is in [project_context.md](project_context.md).

The separate authorized final evaluation on 2026-10-11 also reconciled all
896 reseller orders in [2013-10-01, 2014-04-01) against the clean reference
before scoring labels. October outcomes are now inspected; the source audits
below preserve their original 2026-10-10 scope. See
[final results](evaluation_metrics.md#final-test-evaluation--2026-10-11).

## Backup identity and inspection scope

| Item | Observed value |
| --- | --- |
| Supplied file | `CompanyX.bak` |
| File size | 214,036,480 bytes, approximately 204 MiB |
| SHA-256 | `1a9f1457f96cf2dc00a4ea4bd8212b74f54337a727cd742722bfec4dd0591379` |
| Format | Microsoft SQL Server database backup |
| Logical files | `AdventureWorks2022`, `AdventureWorks2022_log` |
| Audit restoration | Separate local database named `CompanyX_Audit` |
| Audit engine | SQL Server 2022 Developer, 16.0.4295.3 |

The logical file names indicate an AdventureWorks2022 origin. They do not prove
the backup is identical to an unmodified Microsoft sample. In particular, the
supplied data includes invalid reseller orders and nullable detail fields.

The original backup was mounted read-only. Inspection covered schema metadata,
store provenance, aggregate source counts, and reconstruction of orders strictly
before 2013-10-01. No protected final-test churn labels were constructed or
evaluated. Row-count metadata for whole tables is not an evaluation of churn.

## Database structure

The restored database contains 72 user tables across six schemas. Counts below
come from the restored database catalog, not assumptions about a standard sample.

| Schema | Tables | Subject |
| --- | ---: | --- |
| `dbo` | 4 | Database build/logging objects |
| `HumanResources` | 6 | Employees, departments, and employment history |
| `Person` | 13 | People, addresses, and business entities |
| `Production` | 25 | Products, categories, inventory, and production history |
| `Purchasing` | 5 | Vendors and purchasing |
| `Sales` | 19 | Customers, stores, orders, salespeople, and sales reference data |

### Tables relevant to the current datasets

Row counts are catalog counts from the audit restoration. They include all
customers or products where applicable, rather than only reseller orders.

| Table | Rows | Relevant columns and role |
| --- | ---: | --- |
| `Sales.Store` | 701 | `BusinessEntityID`, `Name`, `SalesPersonID`, `Demographics`, `rowguid`, `ModifiedDate`; one store record |
| `Sales.Customer` | 19,820 | `CustomerID`, nullable `PersonID` and `StoreID`, `TerritoryID`, `AccountNumber`; links order customers to resellers |
| `Sales.SalesOrderHeader` | 31,465 | `SalesOrderID`, `CustomerID`, `OrderDate`, `SalesPersonID`, `TerritoryID`, `SubTotal`; one order header |
| `Sales.SalesOrderDetail` | 121,317 | `SalesOrderID`, `SalesOrderDetailID`, `ProductID`, `OrderQty`, `UnitPrice`, `UnitPriceDiscount`, `LineTotal`; one order line |
| `Production.Product` | 504 | `ProductID`, `Name`, `ProductSubcategoryID`; identifies the product for an order line |
| `Production.ProductSubcategory` | 37 | `ProductSubcategoryID`, `ProductCategoryID`, `Name`; links products to categories |
| `Production.ProductCategory` | 4 | `ProductCategoryID`, `Name`; Bikes, Components, Clothing, Accessories |
| `Sales.SalesPerson` | 17 | Salesperson identities; not required to derive the main seven numeric predictors |
| `Sales.SalesTerritory` | 10 | Territory identities; alternative historical categorical features |

`Sales.vStoreWithDemographics` is a view, not another source table. It extracts
store-survey attributes from `Sales.Store.Demographics` XML.

### Joins and field types

The reconstruction uses these keys:

```text
SalesOrderHeader.CustomerID -> Customer.CustomerID
Customer.StoreID -> Store.BusinessEntityID
SalesOrderDetail.SalesOrderID -> SalesOrderHeader.SalesOrderID
SalesOrderDetail.ProductID -> Product.ProductID
Product.ProductSubcategoryID -> ProductSubcategory.ProductSubcategoryID
ProductSubcategory.ProductCategoryID -> ProductCategory.ProductCategoryID
```

The detail-to-product join is used for category aggregation; the database's
declared detail foreign key points to `Sales.SpecialOfferProduct` through the
offer/product pair. Avoid summing header `SubTotal` after joining order lines:
that would repeat the order subtotal once per line.

| Source field | SQL type / availability |
| --- | --- |
| Entity and order IDs | `int`; `Customer.StoreID` is nullable |
| `SalesOrderHeader.OrderDate` | Non-null `datetime` |
| `SalesOrderHeader.SubTotal` | Non-null `money`, four decimal places |
| Header salesperson and territory | Nullable `int` |
| `SalesOrderDetail.OrderQty` | Nullable `smallint` in this backup |
| `SalesOrderDetail.UnitPrice` | Nullable `money` in this backup |
| `SalesOrderDetail.UnitPriceDiscount` | Non-null `money`, four decimal places |
| `SalesOrderDetail.LineTotal` | Computed `numeric(38,6)` |
| `Sales.Store.Demographics` | Nullable `xml` |
| `Sales.Store.ModifiedDate` | Non-null `datetime`; not a field-level availability timestamp |

The computed line revenue is
`ISNULL(UnitPrice * (1 - UnitPriceDiscount) * OrderQty, 0)`.

## Clean order dataset

[orders_clean.csv](../datasets/orders_clean.csv) contains **3,800 rows and 13
columns**, one row per order, with no missing values. The source has 3,806 order
headers linked to a non-null customer `StoreID`.

| CSV column | Meaning and source reconstruction |
| --- | --- |
| `StoreID` | `Sales.Customer.StoreID`, obtained through header `CustomerID` |
| `SalesOrderID` | Header order identifier |
| `OrderDate` | Header order date |
| `TerritoryID` | Header territory at the order |
| `SalesPersonID` | Header salesperson at the order, distinct from the current store salesperson |
| `SubTotal` | Header subtotal, excluding tax and freight |
| `qty` | Sum of detail `OrderQty` per order |
| `n_lines` | Count of detail rows per order |
| `avg_disc` | Unweighted average detail `UnitPriceDiscount` per order |
| `rev_bikes` | Sum of detail `LineTotal` for Bikes |
| `rev_components` | Sum of detail `LineTotal` for Components |
| `rev_clothing` | Sum of detail `LineTotal` for Clothing |
| `rev_accessories` | Sum of detail `LineTotal` for Accessories |

The cleaning contract requires unique order IDs, valid dates, known store IDs,
positive subtotals, and non-null positive quantities. Category revenues must
reconcile with the subtotal within the audit's tolerance. See
[the data-audit stage](../pipeline/01_data_audit/01_audit_source_data.py).

### Direct verification of development orders

Before 2013-10-01, SQL reconstruction yields 2,729 reseller orders, compared with
2,723 in the cleaned CSV. All CSV order keys exist in the source. The six
source-only orders are:

| SalesOrderID | StoreID | Source SubTotal | Additional quantity issue |
| --- | ---: | ---: | --- |
| 43677 | 1044 | 0.0000 | None observed in aggregate quantity |
| 43678 | 1184 | -9799.9243 | None observed in aggregate quantity |
| 43679 | 908 | 0.0000 | None observed in aggregate quantity |
| 43680 | 302 | 0.0000 | Aggregate quantity is null |
| 43681 | 698 | 0.0000 | Aggregate quantity is zero |
| 43682 | 904 | -3875.7229 | Aggregate quantity is negative |

These exclusions agree with the cleaning contract. For all 2,723 matched orders,
IDs, dates, subtotal, territory, salesperson, quantity, line count, and category
revenues match. Discount means agree within `1e-6` absolute tolerance; the maximum
observed difference is approximately `9.804e-7`, consistent with export rounding.
Reconstruction uses `AVG(CAST(UnitPriceDiscount AS float))` to avoid SQL `money`
averaging truncation. The inspected development date range is 2011-05-31 through
2013-09-30.

This is a reconstruction that agrees with the exported data. The original SQL
export script was not available in the current repository. Post-cutoff order
values were not compared in this audit.

## Historical order-line extract for debugging

The later CompanyX feature-discovery scripts expand the catalog inventory to
**491 columns and 91 foreign-key column links** across the same 72 tables.
Saved [tables](../results/debugging/companyx_features/source/tables.csv),
[columns](../results/debugging/companyx_features/source/columns.csv), and
[foreign keys](../results/debugging/companyx_features/source/foreign_keys.csv)
record the inspected schema. These counts describe catalog entries, not the
number of useful churn signals.

For development feature construction, the extraction caps reseller order dates
strictly before **2013-04-01**: 1,892 headers and 28,431 lines. Matching the
established clean-order IDs retains **1,886 headers and 28,356 lines**; the same
six invalid orders remain excluded. Detail keys, completeness, value ranges,
line revenue formulas, and order aggregates are checked before constructing
1,375 historical account/snapshot feature rows. Counts refer to this earlier
cutoff, not to the 2,723 clean orders before October in the source audit above.

The extract keeps quantity, unit price, fractional discount, product identity,
and line revenue. It supports the new
[discount feature definitions](feature_analysis.md#experimental-ten-feature-set)
and other debug candidates. Past order facts are assumed available at order
time; recording-time versions have not been verified.

Fulfillment and current-profile fields do not supply usable historical signals
in this experiment. Accepted order-to-ship duration is always seven days;
ship method and online flag are constant. Current order `Status` has no
historical transition timestamp. Twenty accepted orders have current status 6
(cancelled), but the existing purchase/eligibility/label contract is preserved.
Whether they should count as valid purchases needs a separate source-definition
review rather than silently changing labels. Current mutable Store profiles
remain excluded under the provenance limits below.

The metadata review catalogued product price/cost histories and salesperson
territory history, but these were not extracted or evaluated. ProductReview has
four records across three products without a direct reseller key; no linked
account service or payment-failure history was established from the reviewed
sources. This does not rule out useful signals in other CompanyX systems.
Card credentials were not extracted.

The [cleaning manifest](../results/debugging/companyx_features/cleaning_manifest.json)
records counts and input hashes. Full retrieval/cleaning findings are in
[debugging.md](debugging.md#companyx-feature-discovery-and-usefulness-experiment--2026-10-10);
commands and prerequisites are in the
[pipeline guide](../pipeline/README.md#debugging-and-feature-experiments).

## Clean store dataset

[stores_clean.csv](../datasets/stores_clean.csv) contains **701 rows and 11
columns**, one row per store, with no missing values. All rows and exported
fields match the restored source.

| CSV column | Source | Meaning |
| --- | --- | --- |
| `StoreID` | `Sales.Store.BusinessEntityID` | Reseller identity |
| `SalesPersonID` | `Sales.Store.SalesPersonID` | Store-profile salesperson |
| `AnnualSales` | Survey XML, exposed by the demographics view as `money` | Store survey's annual-sales value |
| `AnnualRevenue` | Survey XML, exposed as `money` | Store survey's annual-revenue value |
| `BusinessType` | Survey XML | Business classification |
| `YearOpened` | Survey XML, exposed as integer | Store opening year |
| `Specialty` | Survey XML | Store specialty |
| `SquareFeet` | Survey XML | Store area |
| `Brands` | Survey XML | Brand classification |
| `Internet` | Survey XML | Internet classification |
| `NumberEmployees` | Survey XML | Employee count |

The view uses XML namespace
`http://schemas.microsoft.com/sqlserver/2004/07/adventure-works/StoreSurvey`.
Survey fields such as annual revenue are not reconstructed sums of this
project's historical order windows.

### Opening-year assumption and provenance limits

The project assumes `YearOpened` describes a fixed historical fact known at
each prediction date. Consequently, `store_age = snapshot.year - YearOpened`
is eligible as a historical feature under that explicit assumption. A store
opened in 2000 has a calendar-year age of 13 at a 2013 snapshot; its exact opening
day is unavailable.

The backup confirms opening years of 1970–2001, all before the development
snapshots. All 701 opening-year values match the CSV. `Sales.Store` is not a
temporal table, has no CDC enabled, and has no store-table triggers. All stored
`ModifiedDate` values are `2014-09-12 11:15:07.497`. These facts do not establish
when each opening year was first recorded. We verified source values, not
historical recording dates. No future-information leakage from `store_age` has
been demonstrated.

This assumption does not certify changing profile attributes such as annual
revenue, employee count, or assigned salesperson. Those remain optional
diagnostic inputs until their historical availability is established. Existing
history tables for other subjects do not supply opening-year history by
themselves.

## Processed references

| Tracked file | Rows | Columns | Role |
| --- | ---: | ---: | --- |
| `datasets/processed/reseller_order_age_features.csv` | 2,312 | 19 | Six snapshots; 17 predictors plus store ID and snapshot |
| `datasets/processed/ml_labeled_snapshots.csv` | 1,375 | 20 | Four development snapshots, adding `churn` |
| `datasets/processed/ml_test_features.csv` | 492 | 19 | Protected 2013-10-01 features without churn labels |

Missing purchase-gap and revenue-trend values can be meaningful: a reseller may
have insufficient distinct order dates or no earlier six-month revenue. They
are not automatically invalid source rows. See
[feature_analysis.md](feature_analysis.md) for formulas and treatment.

These CSVs are regression references, not disposable generated reports. Preserve
them when changing documentation or conducting source inspection.

## Repeating the metadata inspection

After restoring an audit database, these queries reproduce relevant catalog
counts and opening-year provenance without constructing churn labels:

```sql
USE CompanyX_Audit;
SET QUOTED_IDENTIFIER ON;

SELECT s.name AS schema_name, t.name AS table_name, SUM(p.rows) AS metadata_rows
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
JOIN sys.partitions AS p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
GROUP BY s.name, t.name
ORDER BY s.name, t.name;

SELECT OBJECT_DEFINITION(OBJECT_ID('Sales.vStoreWithDemographics'));

SELECT COUNT(*) AS stores, MIN(YearOpened) AS earliest_year,
       MAX(YearOpened) AS latest_year
FROM Sales.vStoreWithDemographics;

SELECT MIN(ModifiedDate) AS earliest_modified,
       MAX(ModifiedDate) AS latest_modified
FROM Sales.Store;

SELECT name, temporal_type_desc, is_tracked_by_cdc
FROM sys.tables WHERE name = 'Store';
```

"""Extract historical order facts and source profiles; never construct labels."""

from pathlib import Path

from companyx_readonly import query


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/debugging/companyx_features/source"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    headers = query("""
SELECT c.StoreID,h.SalesOrderID,
       CONVERT(varchar(23),h.OrderDate,121) AS OrderDate,
       CONVERT(varchar(23),h.DueDate,121) AS DueDate,
       CONVERT(varchar(23),h.ShipDate,121) AS ShipDate,
       h.SubTotal,h.TerritoryID,h.SalesPersonID,h.ShipMethodID,
       h.Status,h.OnlineOrderFlag,h.TaxAmt,h.Freight
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
ORDER BY h.SalesOrderID;
""")
    lines = query("""
SELECT h.SalesOrderID,d.SalesOrderDetailID,d.ProductID,d.SpecialOfferID,
       d.OrderQty,d.UnitPrice,d.UnitPriceDiscount,d.LineTotal
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
JOIN Sales.SalesOrderDetail d ON d.SalesOrderID=h.SalesOrderID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
ORDER BY h.SalesOrderID,d.SalesOrderDetailID;
""")
    profiles = query("""
SELECT 'reseller_order_history_before_2013_04' AS subject,
       COUNT(*) AS rows,
       COUNT(DISTINCT h.Status) AS distinct_values,
       'Status' AS profiled_field
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
UNION ALL
SELECT 'reseller_order_history_before_2013_04',COUNT(*),
       COUNT(DISTINCT DATEDIFF(day,h.OrderDate,h.ShipDate)), 'days_to_ship'
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
UNION ALL
SELECT 'reseller_order_history_before_2013_04',COUNT(*),
       COUNT(DISTINCT DATEDIFF(day,h.OrderDate,h.DueDate)), 'scheduled_days'
FROM Sales.SalesOrderHeader h JOIN Sales.Customer c ON c.CustomerID=h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
UNION ALL
SELECT 'ProductReview',COUNT(*),COUNT(DISTINCT ProductID),'ProductID'
FROM Production.ProductReview
UNION ALL
SELECT 'Store',COUNT(*),COUNT(DISTINCT ModifiedDate),'ModifiedDate'
FROM Sales.Store;
""")
    for name, frame in [("order_headers_raw", headers), ("order_lines_raw", lines), ("source_profiles", profiles)]:
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(f"Extracted {len(headers)} reseller order headers and {len(lines)} lines before 2013-04-01")
    print(profiles.to_string(index=False))
    for column in ["Status", "OnlineOrderFlag", "ShipMethodID"]:
        print(column, headers[column].value_counts(dropna=False).to_dict())
    print('Discount unique values:', sorted(lines['UnitPriceDiscount'].dropna().unique()))
    print('Products:', lines['ProductID'].nunique())


if __name__ == "__main__":
    main()

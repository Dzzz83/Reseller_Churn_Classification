-- Optional SQL Server input extraction. Export each result set as a CSV with
-- headers. Set the cutoff to the last required feature/label boundary; no row
-- at or after the cutoff is included. Credentials and database setup are local.
DECLARE @CutoffDate date = '2013-10-01';

SELECT c.StoreID, h.SalesOrderID,
       CONVERT(varchar(23), h.OrderDate, 121) AS OrderDate, h.SubTotal
FROM Sales.SalesOrderHeader AS h
JOIN Sales.Customer AS c ON c.CustomerID = h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < @CutoffDate
ORDER BY h.SalesOrderID;

SELECT d.SalesOrderID, d.SalesOrderDetailID, d.ProductID,
       d.OrderQty, d.UnitPrice, d.UnitPriceDiscount, d.LineTotal
FROM Sales.SalesOrderDetail AS d
JOIN Sales.SalesOrderHeader AS h ON h.SalesOrderID = d.SalesOrderID
JOIN Sales.Customer AS c ON c.CustomerID = h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < @CutoffDate
ORDER BY d.SalesOrderID, d.SalesOrderDetailID;

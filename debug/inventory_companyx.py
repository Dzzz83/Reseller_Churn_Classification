"""Save CompanyX schema inventory and historical-availability candidates."""

from pathlib import Path

from companyx_readonly import query


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/debugging/companyx_features/source"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = query("""
SELECT s.name AS schema_name, t.name AS table_name,
       SUM(p.rows) AS metadata_rows, t.temporal_type_desc, t.is_tracked_by_cdc
FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
JOIN sys.partitions p ON p.object_id=t.object_id AND p.index_id IN (0,1)
GROUP BY s.name,t.name,t.temporal_type_desc,t.is_tracked_by_cdc
ORDER BY s.name,t.name;
""")
    columns = query("""
SELECT s.name AS schema_name, t.name AS table_name, c.column_id, c.name AS column_name,
       ty.name AS data_type, c.max_length, c.precision, c.scale, c.is_nullable,
       c.is_computed
FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
JOIN sys.columns c ON c.object_id=t.object_id
JOIN sys.types ty ON ty.user_type_id=c.user_type_id
ORDER BY s.name,t.name,c.column_id;
""")
    keys = query("""
SELECT fk.name AS foreign_key,
       OBJECT_SCHEMA_NAME(f.parent_object_id) AS source_schema,
       OBJECT_NAME(f.parent_object_id) AS source_table,
       COL_NAME(f.parent_object_id,f.parent_column_id) AS source_column,
       OBJECT_SCHEMA_NAME(f.referenced_object_id) AS target_schema,
       OBJECT_NAME(f.referenced_object_id) AS target_table,
       COL_NAME(f.referenced_object_id,f.referenced_column_id) AS target_column
FROM sys.foreign_key_columns f
JOIN sys.foreign_keys fk ON fk.object_id=f.constraint_object_id
ORDER BY source_schema,source_table,foreign_key,f.constraint_column_id;
""")
    descriptions = query("""
SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name,
       CAST(e.value AS nvarchar(1500)) AS description
FROM sys.extended_properties e
JOIN sys.tables t ON t.object_id=e.major_id
JOIN sys.schemas s ON s.schema_id=t.schema_id
JOIN sys.columns c ON c.object_id=t.object_id AND c.column_id=e.minor_id
WHERE e.class=1 AND e.name='MS_Description'
  AND t.name IN ('SalesOrderHeader','SalesOrderDetail')
ORDER BY s.name,t.name,c.column_id;
""")
    for name, table in [
        ("tables", tables), ("columns", columns), ("foreign_keys", keys),
        ("column_descriptions", descriptions),
    ]:
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    relevant = columns[
        columns["table_name"].isin([
            "SalesOrderHeader", "SalesOrderDetail", "Store", "ProductReview",
            "ProductListPriceHistory", "ProductCostHistory", "TransactionHistory",
            "Customer", "SalesPerson", "SalesTerritoryHistory",
        ])
    ]
    print(f"Inventory: {len(tables)} tables, {len(columns)} columns, {len(keys)} foreign-key column links")
    print(relevant[["schema_name", "table_name", "column_name", "data_type"]].to_string(index=False))


if __name__ == "__main__":
    main()

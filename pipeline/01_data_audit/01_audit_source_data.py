from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.source_audit import audit_source_data


def main() -> None:
    print("=== 01. Data Audit ===")

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()

    audit_source_data(orders, stores)

    print(f"[PASS] Orders: {len(orders):,} rows")
    print(f"[PASS] Stores: {len(stores):,} rows")
    print(
        "[PASS] IDs, dates, quantities, revenue, store joins, "
        "and category revenue are valid."
    )


if __name__ == "__main__":
    main()

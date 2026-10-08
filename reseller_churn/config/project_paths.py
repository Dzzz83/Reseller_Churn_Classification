from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

ORDERS_PATH = DATASETS_DIR / "orders_clean.csv"
STORES_PATH = DATASETS_DIR / "stores_clean.csv"
LABELED_SNAPSHOTS_PATH = PROCESSED_DATA_DIR / "ml_labeled_snapshots.csv"
FINAL_TEST_FEATURES_PATH = PROCESSED_DATA_DIR / "ml_test_features.csv"
ALL_ENGINEERED_FEATURES_PATH = PROCESSED_DATA_DIR / "reseller_order_age_features.csv"

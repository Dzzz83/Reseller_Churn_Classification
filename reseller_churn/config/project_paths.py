import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Compatibility scripts write generated preparation files here. The verified
# CSVs below remain read-only regression references.
LEGACY_PREPARATION_DIR = RESULTS_DIR / "legacy_preparation"
LEGACY_ENGINEERED_FEATURES_PATH = (
    LEGACY_PREPARATION_DIR / "reseller_order_age_features.csv"
)
LEGACY_LABELED_SNAPSHOTS_PATH = LEGACY_PREPARATION_DIR / "ml_labeled_snapshots.csv"
LEGACY_FINAL_TEST_FEATURES_PATH = LEGACY_PREPARATION_DIR / "ml_test_features.csv"

ORDERS_PATH = DATASETS_DIR / "orders_clean.csv"
STORES_PATH = DATASETS_DIR / "stores_clean.csv"
LABELED_SNAPSHOTS_PATH = PROCESSED_DATA_DIR / "ml_labeled_snapshots.csv"
FINAL_TEST_FEATURES_PATH = PROCESSED_DATA_DIR / "ml_test_features.csv"
ALL_ENGINEERED_FEATURES_PATH = PROCESSED_DATA_DIR / "reseller_order_age_features.csv"


def create_run_id() -> str:
    """Return a filesystem-safe UTC identifier for one generated pipeline run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def get_run_directory(run_id: str) -> Path:
    """Resolve a run ID under results without allowing path traversal."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError(
            "run_id must be 1-80 filesystem-safe letters, numbers, dots, "
            "underscores, or hyphens."
        )
    return RESULTS_DIR / run_id

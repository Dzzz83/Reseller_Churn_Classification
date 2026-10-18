"""Prepare leakage-safe development datasets under a new results run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.artifacts import file_sha256, write_json
from reseller_churn.config.project_paths import (
    ALL_ENGINEERED_FEATURES_PATH,
    FINAL_TEST_FEATURES_PATH,
    LABELED_SNAPSHOTS_PATH,
    ORDERS_PATH,
    STORES_PATH,
    create_run_id,
    get_run_directory,
)
from reseller_churn.config.validation_settings import ALL_FEATURE_SNAPSHOTS
from reseller_churn.data.dataset_loader import DatasetLoader
from reseller_churn.data.ml_dataset_builder import build_ml_datasets
from reseller_churn.data.source_audit import audit_source_data
from reseller_churn.evaluation.regression_checks import RegressionChecks
from reseller_churn.features.feature_pipeline import FeaturePipeline
from reseller_churn.modeling.development import repository_identity


def prepare_data(run_id: str) -> Path:
    """Generate and regression-check data without writing into tracked references."""
    run_directory = get_run_directory(run_id)
    if run_directory.exists():
        raise FileExistsError(
            f"Run directory already exists; choose a new run ID: {run_directory}"
        )

    preparation_directory = run_directory / "preparation"
    development_directory = run_directory / "development"
    evaluation_directory = run_directory / "evaluation"

    orders = DatasetLoader.load_orders()
    stores = DatasetLoader.load_stores()
    source_audit = audit_source_data(orders, stores)
    engineered_features = FeaturePipeline().build(
        orders=orders,
        stores=stores,
        snapshots=ALL_FEATURE_SNAPSHOTS,
    )
    reference_features = pd.read_csv(
        ALL_ENGINEERED_FEATURES_PATH,
        parse_dates=["snapshot"],
    )
    RegressionChecks.assert_feature_frames_match(
        expected=reference_features,
        actual=engineered_features,
    )

    labeled_data, final_test_features = build_ml_datasets(
        orders=orders,
        engineered_features=engineered_features,
    )
    for reference_path, actual in (
        (LABELED_SNAPSHOTS_PATH, labeled_data),
        (FINAL_TEST_FEATURES_PATH, final_test_features),
    ):
        reference = pd.read_csv(reference_path, parse_dates=["snapshot"])
        RegressionChecks.assert_feature_frames_match(
            expected=reference,
            actual=actual,
        )

    preparation_directory.mkdir(parents=True)
    development_directory.mkdir()
    evaluation_directory.mkdir()
    feature_path = preparation_directory / "engineered_features.csv"
    labeled_path = preparation_directory / "labeled_development.csv"
    final_features_path = preparation_directory / "final_test_features_unlabeled.csv"
    source_audit_path = preparation_directory / "source_audit.json"
    engineered_features.to_csv(feature_path, index=False)
    labeled_data.to_csv(labeled_path, index=False)
    final_test_features.to_csv(final_features_path, index=False)
    write_json(
        source_audit_path,
        {
            **source_audit,
            "source_files": {
                "orders": {
                    "path": ORDERS_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": file_sha256(ORDERS_PATH),
                },
                "stores": {
                    "path": STORES_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": file_sha256(STORES_PATH),
                },
            },
        },
    )

    write_json(
        preparation_directory / "preparation_manifest.json",
        {
            "run_id": run_id,
            "outputs": {
                "engineered_features": feature_path.name,
                "labeled_development": labeled_path.name,
                "final_test_features": final_features_path.name,
                "source_audit": source_audit_path.name,
            },
            "feature_snapshots": list(ALL_FEATURE_SNAPSHOTS),
            "development_rows": len(labeled_data),
            "development_churners": int(labeled_data["churn"].sum()),
            "final_test_feature_rows": len(final_test_features),
            "final_test_labels_accessed": False,
            "reference_checks": {
                "engineered_features": "passed",
                "labeled_development": "passed",
                "final_test_features": "passed",
            },
            "repository": repository_identity(),
        },
    )
    (evaluation_directory / "STATUS.md").write_text(
        "# Protected evaluation status\n\n"
        "No final evaluation was run. The final-test feature rows are unlabeled, "
        "and final-test labels were not accessed. Follow the guarded instructions "
        "in `pipeline/12_final_evaluation/README.md` before any final evaluation.\n",
        encoding="utf-8",
    )

    print(f"Prepared data under: {preparation_directory}")
    print(f"Development rows: {len(labeled_data)}")
    print(f"Final-test features only: {len(final_test_features)}")
    return labeled_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    run_id = arguments.run_id or create_run_id()
    prepare_data(run_id)
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()

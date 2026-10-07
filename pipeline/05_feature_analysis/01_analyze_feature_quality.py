from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from reseller_churn.step_00_config.feature_sets import FULL_FEATURES
from reseller_churn.step_00_config.project_paths import RESULTS_DIR
from reseller_churn.step_01_data.dataset_loader import DatasetLoader


OUTPUT_DIR = RESULTS_DIR / "feature_analysis"


def main() -> None:
    print("=== 05.1 Feature Quality Analysis ===")

    data = DatasetLoader.load_labeled_snapshots()

    missingness = (
        data[FULL_FEATURES]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index(name="feature")
    )

    correlations = (
        data[
            FULL_FEATURES + ["churn"]
        ]
        .corr(numeric_only=True)["churn"]
        .drop("churn")
        .rename("correlation_with_churn")
        .reset_index(name="feature")
    )

    summary = (
        missingness.merge(
            correlations,
            on="feature",
            validate="one_to_one",
        )
        .assign(
            absolute_correlation=lambda frame:
                frame["correlation_with_churn"].abs()
        )
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "01_feature_quality_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )
    print()
    print(f"Saved: {output_path}")
    print("Final-test labels were NOT used.")


if __name__ == "__main__":
    main()

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_STEPS = [
    "pipeline/01_data_audit/01_audit_source_data.py",
    "pipeline/02_problem_definition/01_verify_prediction_windows.py",
    "pipeline/02_problem_definition/02_verify_churn_labels.py",
    "pipeline/03_feature_engineering/01_build_all_features.py",
    "pipeline/04_ml_dataset_assembly/01_build_ml_datasets.py",
    "pipeline/05_feature_analysis/01_analyze_feature_quality.py",
    "pipeline/05_feature_analysis/02_analyze_fold2_errors.py",
    "pipeline/06_model_selection/01_compare_model_architectures.py",
    "pipeline/07_feature_selection/01_compare_feature_sets.py",
    "pipeline/08_imbalance_strategy/01_compare_imbalance_strategies.py",
    "pipeline/10_threshold_selection/01_select_thresholds.py",
    "pipeline/11_provisional_training/01_train_provisional_models.py",
    "pipeline/11_provisional_training/02_evaluate_development_models.py",
]


def main() -> None:
    print(
        "=== Full Development Pipeline ===",
        flush=True,
    )
    print(
        "Stage 12 final evaluation is intentionally excluded.",
        flush=True,
    )
    print()

    for index, relative_path in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        script_path = (
            PROJECT_ROOT
            / relative_path
        )

        print(
            "=" * 80,
            flush=True,
        )
        print(
            f"[{index}/{len(PIPELINE_STEPS)}] "
            f"{relative_path}",
            flush=True,
        )
        print(
            "=" * 80
        )

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        print()

    print(
        "=== Development Pipeline Completed ==="
    )
    print(
        "Final-test labels were NOT used."
    )


if __name__ == "__main__":
    main()

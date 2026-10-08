# Reseller Churn Classification

Predict whether an **active reseller** will place no orders during the six
months after a historical prediction snapshot.

The codebase is intentionally organized for human readability. Folder and file
names show the order of the pipeline, and reusable implementation is separated
from experiment orchestration.

## Repository Structure

```text
reseller_churn/
├── config/       shared paths, feature sets, folds, model settings
├── data/         windows, eligibility, labels, temporal datasets
├── features/     leakage-safe feature engineering
├── modeling/     model factory, resampling, seeded predictions
└── evaluation/   metrics, thresholds, regression checks

pipeline/
├── 01_data_audit/
├── 02_problem_definition/
├── 03_feature_engineering/
├── 04_ml_dataset_assembly/
├── 05_feature_analysis/
├── 06_model_selection/
├── 07_feature_selection/
├── 08_imbalance_strategy/
├── 09_hyperparameter_tuning/
├── 10_threshold_selection/
├── 11_provisional_training/
└── 12_final_evaluation/

tests/
├── data/
├── features/
└── modeling/
```

## Pipeline Order

To run the complete development pipeline through provisional model evaluation:

```bash
python pipeline/run_full_development_pipeline.py
```

This deliberately stops before Stage 12 and does not evaluate the protected
2013-10-01 final test.

To run individual stages, use the commands below from the repository root.


```bash
python pipeline/01_data_audit/01_audit_source_data.py
python pipeline/02_problem_definition/01_verify_prediction_windows.py
python pipeline/02_problem_definition/02_verify_churn_labels.py
python pipeline/03_feature_engineering/01_build_all_features.py
python pipeline/04_ml_dataset_assembly/01_build_ml_datasets.py
python pipeline/05_feature_analysis/01_analyze_feature_quality.py
python pipeline/05_feature_analysis/02_analyze_fold2_errors.py
python pipeline/06_model_selection/01_compare_model_architectures.py
python pipeline/07_feature_selection/01_compare_feature_sets.py
python pipeline/08_imbalance_strategy/01_compare_imbalance_strategies.py
# Optional research-only exhaustive retuning:
# python pipeline/09_hyperparameter_tuning/01_tune_models.py
python pipeline/10_threshold_selection/01_select_thresholds.py
python pipeline/11_provisional_training/01_train_provisional_models.py
```

Stage 12 is intentionally guarded and has no executable final-test script yet.

## Anti-Leakage Rules

At snapshot `t`:

- eligibility uses only `[t - 6 months, t)`;
- features use only historical data before `t`;
- the churn label uses only `[t, t + 6 months)`;
- any learned preprocessing or model choice uses development data only;
- the 2013-10-01 final test remains untouched until all choices are frozen.

## Development Folds

```text
Fold 1
Train:      2012-07-01
Validation: 2013-01-01

Fold 2
Train:      2012-07-01 + 2012-10-01
Validation: 2013-04-01
```

## Current Locked Model Inputs

The current pruned feature set is:

```text
n_orders_3m
revenue_3m
recency_days
share_bikes
share_accessories
share_clothing
revenue_12m
```

Current tuned model settings and seeds live only in:

```text
reseller_churn/config/model_settings.py
```

That file is the single source of truth.

## Validation of the Refactor

The tests protect the business rules and previously verified results:

```bash
pytest
```

They check:

- prediction-window boundaries;
- historical churn counts;
- feature parity with the verified engineered dataset;
- exact Stage-08 Random Forest and Gradient Boosting PR-AUC baselines.

The purpose of the refactor is structural clarity. It must not silently change
the validated methodology or model results.

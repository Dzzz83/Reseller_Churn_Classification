# Pipeline Execution Order

The folder and file names are intentionally numbered so the repository shows
the project methodology in execution order.

1. **01_data_audit** — verify cleaned source data.
2. **02_problem_definition** — verify prediction windows, eligibility, and churn labels.
3. **03_feature_engineering** — build historical reseller features.
4. **04_ml_dataset_assembly** — combine historical features with development labels and keep the final test unlabeled.
5. **05_feature_analysis** — inspect feature behavior and model failure modes.
6. **06_model_selection** — compare model architectures.
7. **07_feature_selection** — compare feature sets.
8. **08_imbalance_strategy** — compare imbalance handling.
9. **09_hyperparameter_tuning** — tune the finalist models.
10. **10_threshold_selection** — select classification thresholds on development data.
11. **11_provisional_training** — train provisional models using frozen development choices.
12. **12_final_evaluation** — reserved for the single final-test evaluation after every choice is frozen.

Reusable implementation lives in the matching ordered package:

```text
reseller_churn/
├── config/
├── data/
├── features/
├── modeling/
└── evaluation/
```

The numbered pipeline files are orchestration only. Business rules and reusable
ML logic belong in `reseller_churn/`.

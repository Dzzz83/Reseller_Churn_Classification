# Development Pipeline

Numbered directories show the **execution order** of experiments. Reusable
implementation is kept in the unnumbered `reseller_churn/` package.

| Stage | Purpose |
| --- | --- |
| 01 Data audit | Verify cleaned source data |
| 02 Problem definition | Verify time windows, eligibility and labels |
| 03 Feature engineering | Build past-only reseller features |
| 04 ML dataset assembly | Build labeled development rows and unlabeled final-test features |
| 05 Feature analysis | Analyze features and fold-2 classification errors |
| 06 Model selection | Compare architectures; model-design-factor audit is optional |
| 07 Feature selection | Compare predefined feature sets |
| 08 Imbalance strategy | Compare class imbalance treatments |
| 09 Hyperparameter tuning | Optional exhaustive development-only search |
| 10 Threshold selection | Analyze provisional thresholds on development data |
| 11 Provisional training | Train and evaluate development models |
| 12 Final evaluation | Intentionally guarded; not executable |

Run from the repository root:

```bash
python pipeline/run_full_development_pipeline.py
```

The runner executes the numbered stages except **09** (optional), **12**
(protected final evaluation), and the optional additional model-design-factor
comparison in Stage 06.

All labels and evaluation metrics in the runner belong to development
snapshots. The protected final-test features may be assembled, but **final-test
outcomes are never accessed**.

Additional research-only commands:

```bash
python pipeline/06_model_selection/02_compare_model_design_factors.py
python comparisons/validation_schedule_audit.py
python pipeline/09_hyperparameter_tuning/01_tune_models.py
```

Generated experiment reports go to `results/`, and provisional model objects
go to `models/provisional/`. These are reproducible outputs and are ignored by
Git. See the root README for the regression references, evaluation policies,
and the two different validation questions.

The schedule audit tests whether a complete 12-month lookback and two resolved
six-month label windows can support selecting both validation procedures **before**
a shared development holdout. With orders starting 2011-05-31 and a protected
final-test snapshot of 2013-10-01, the earliest temporal results arrive after
the latest permitted development holdout. This is a time-coverage constraint,
not evidence that either validation method has better predictive performance.

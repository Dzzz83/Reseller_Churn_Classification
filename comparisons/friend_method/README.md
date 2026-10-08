# Friend-method verification (isolated)

Original source: https://github.com/pnn-re1506/CurrentTopicTest

This is a **corrected reconstruction**, not an edit to your teammate's
notebooks. It does not alter the primary reseller churn pipeline.

## Problems and corrections

| Source behavior | Correction |
| --- | --- |
| Uses 3,806 raw orders, including six invalid records | Reads the verified 3,800-order clean input |
| Uses `OrderDate <= snapshot + 6 months` | Uses half-open `[snapshot, snapshot + 6 months)` |
| Uses `GroupKFold(StoreID)` for primary CV, mixing calendar dates | Uses two past-to-future folds with resolved training labels |
| Selects features using all CV training labels before splitting | Recomputes supervised correlation pruning separately within every training fold |
| Uses current `AnnualSales`, `BusinessType`, `Specialty`, `Brands`, `Internet`, `store_salesperson` | Excludes all historically unverifiable store fields; keeps stable `store_age` |
| Uses potentially historical order-level territory | Keeps historical last-order `TerritoryID`; obtains salesperson from the last historical order rather than the current store record |
| Calls all-before-snapshot features 'full' | Names this variant `full`; `obs6` means just the previous six months |
| Selects F1 threshold with OOF labels | Threshold selection is deferred until development model selection is decided; 0.5 metrics are diagnostic only |

The implementations keep your teammate's original modeling family:
RandomForest (500 trees, min leaf 5) and LogisticRegression, using
numeric log/median/indicator/scale preprocessing and categorical one-hot
encoding. Correlated numeric features are pruned with Spearman rho > 0.8
using each fold's training rows.

## Protected timeline

The primary project has a protected final test on **2013-10-01**.
The 2013-05-01 teammate snapshot has labels ending on 2013-11-01,
so those outcomes were **not known at the October prediction date**.

Therefore this development audit restricts snapshots to:

- 2012-05-01
- 2012-08-01
- 2012-11-01
- 2013-02-01

Development folds:

1. Train 2012-05 -> validate 2012-11
2. Train 2012-05, 2012-08 -> validate 2013-02

The final validation snapshot 2013-02 has labels ending 2013-08,
which is before the October test date. Both feature variants use the
same eligible resellers and labels.

## Run (from repository root)

```bash
python -m pytest -q
python comparisons/friend_method/01_verify_development.py
```

Outputs are written to `results/friend_method_audit/`:

- `01_development_fold_metrics.csv`
- `01_development_model_summary.csv`
- `01_development_predictions.csv`

**The script does not construct November 2013 outcomes or evaluate
October 2013 test labels.**

## Important methodological caveats

- Because this corrects data quality and excludes unverifiable features,
  it is **not an exact score reproduction** of the friend's original
  0.500 CV / 0.518 test. The aim is a fair and deployable variant.
- The friend's November 2013 test has already been examined, so it should
  not be treated as a pristine untouched benchmark for iterative tuning.
- Selecting improvements based on test outcomes after October 2013
  would compromise the protected October test.
- A post-freeze evaluation of the friend's November snapshot requires
  an explicit independent audit stage with those caveats stated.
- Per-fold F1 at 0.5 is **not** a selected operational threshold.
- Temporal validation can reuse a StoreID across time. That is valid
  for forecasting existing resellers and differs from evaluating unseen IDs.
- These CV results are development estimates and can be optimistic after
  repeated model decisions. Avoid claiming statistical certainty from
  only two temporal folds.

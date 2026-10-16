# 12. Final Evaluation — Guarded Stage

The user-authorized October evaluation completed on 2026-10-11 through the
guarded entry point in root `debug/`. The supported development runner still
does not execute final evaluation automatically.

The final-test snapshot is **2013-10-01**. Before accessing its labels, the
following requirements were satisfied:

1. feature set is frozen;
2. model architecture is frozen;
3. imbalance strategy is frozen;
4. hyperparameters are frozen;
5. threshold strategy is frozen;
6. the user explicitly approves protected final evaluation.

The frozen methodology is ten-feature histogram gradient boosting, existing
baseline parameters, balanced training weights, native missing handling, and
the average of seeds 42, 78, 88, 1034, 2026. The user selected grouped-development
cutoffs: 30% precision floor as primary (0.06), plus predeclared 20% (0.01),
40% (0.55), and 50% (0.88) secondary rules.

## Completed entry point and artifacts

[run_final_testing_evaluation.py](../../debug/run_final_testing_evaluation.py)
has separate `--stage prepare` and `--stage evaluate` commands. Preparation saves
the protocol, refits five models on the 1,375 approved development rows, and
checkpoints 492 unlabeled test predictions. Evaluation verifies code/data/model
and prediction hashes before source reconciliation and label construction;
it fits no models and does not change thresholds. Existing preparations and
completed evaluations cannot be overwritten through the entry point.

The completed report is under
`results/final_evaluation/gb10_grouped_20261011/`. Its primary result is **93.48%
recall, 28.48% precision, and F1 0.437**, catching **43 of 46 churners**, missing
three, and producing **108 false alarms**. The primary
rule misses the intended 30% precision requirement. See
[final metrics and limitations](../../docs/evaluation_metrics.md#final-test-evaluation--2026-10-11)
and [the full record](../../docs/debugging.md#final-testing-evaluation--2026-10-11).

The outcome window is [2013-10-01, 2014-04-01), with 46 churners among 492
accounts. Outcomes are **now inspected**. Do not treat rerunning this cohort or
using its results to choose new parameters/cutoffs as another untouched test.
Development guards continue to exclude these outcomes. A future independent
evaluation requires an appropriate new holdout and frozen methodology.

Verification of the completed artifacts does not refit or reveal new outcomes:

```bash
bi_venv/bin/python debug/verify_final_testing_outputs.py
```

# Frozen model and original evaluation

The final pipeline carries the methodology frozen before the original
2026-10-11 evaluation. Its source research commit is `f53202d` on
`pipeline_preparation`. Parameter/feature/seed changes require explicit new
methodology; probability-cutoff comparisons do not alter the trained model.

- Histogram gradient boosting: learning_rate=0.1, max_iter=100,
  max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0,
  early_stopping=False.
- Balanced training sample weights, native missing-value handling.
- Mean predicted probabilities across seeds 42, 78, 88, 1034, 2026.
- Ten features in the order recorded by the model manifest and input contract.
- Training snapshots July/October 2012 and January/April 2013: 1,375 rows,
  including 212 churn cases in the original dataset.
- Original primary cutoff: 0.06, selected from grouped development results.
  No model search or threshold selection is performed by the final pipeline.

The original test snapshot is 2013-10-01, with outcomes in
`[2013-10-01, 2014-04-01)`: 492 accounts, 46 churners. Predictions and fixed
alert decisions preceded outcome access. The original report remains under
`results/final_evaluation/gb10_grouped_20261011/` in the research checkout;
generated files and the original model remain outside Git.

| Cutoff | Original role | Recall | Precision | Caught | Missed | False alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | Secondary | 97.83% | 20.64% | 45 | 1 | 173 |
| 0.06 | Primary | 93.48% | 28.48% | 43 | 3 | 108 |
| 0.55 | Secondary | 71.74% | 56.90% | 33 | 13 | 25 |
| 0.88 | Secondary | 43.48% | 76.92% | 20 | 26 | 6 |

The primary rule fell below its intended 30% precision floor. Recently active
accounts had 8 caught churners, 3 misses, and 98 false alerts: 72.73% recall and
7.55% precision. Previously inactive accounts contained 35 of the 46 churners,
all caught by the primary rule. This subgroup mix limits interpretation of
overall recall as stable future-period performance.

The model remains provisional in terms of operational suitability. Merging a
tested implementation does not resolve alert-cost requirements, subgroup
weaknesses, probability calibration, or the source availability assumptions.
Changing cutoffs using these inspected outcomes requires a new independent
holdout to validate fresh performance. Preserve the original record and avoid
calling subsequent comparisons another untouched final test.

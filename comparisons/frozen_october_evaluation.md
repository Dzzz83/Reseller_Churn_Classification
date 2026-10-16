# Frozen candidate: inspected October evaluation

On 2026-10-16 the user explicitly requested evaluation using the final test set.
The already frozen 13-feature cadence/regularization candidate is scored without
training or cutoff changes. October outcomes were already inspected before this
follow-up; this is descriptive reuse, not a new independent final test.

## Results

The cohort has 492 resellers and 46 churners. Scoring is at 2013-10-01, with
outcomes in `[2013-10-01,2014-04-01)`. The candidate preserves the original ten
features plus overdue ratio, annual order count, and gap variation; seven
leaves, L2=1, five registered model seeds averaged, native missing handling,
training-only class weights, and cutoff `0.21655335464238584`.

| Rule | Recall | Precision | Caught | Missed | False alerts | Alerts | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, cutoff 0.06 | 93.48% | 28.48% | 43 | 3 | 108 | 151 | 0.7013 |
| Frozen candidate, cutoff 0.2165533546 | 97.83% | 20.09% | 45 | 1 | 179 | 224 | 0.7892 |

Precision decreases **8.39 percentage points** and recall increases **4.35
points**. The two extra caught churners cost 71 net additional false alerts.
Paired changes are two newly caught, none newly missed, ten false alerts
removed, and 81 added. **The candidate fails the precision-improvement objective
on this cohort.** No other cutoff or candidate is selected from these outcomes.

Both rules catch all 35 inactive churners with ten false alerts. Among active
accounts, the candidate catches ten of 11 with 169 false alerts, versus eight
with 98 false alerts. Active recall rises from 72.73% to 90.91%, but precision
falls from 7.55% to 5.59%. Active AP falls from 0.1553 to 0.0765, while inactive
AP rises from 0.8870 to 0.9469. Aggregate AP improvement does not establish
better active-account discrimination or a better frozen operating point.

## Provenance and execution

The saved bundle/specification are under `results/repeated-grouped-precision/`.
Unlabeled features and supplied outcomes come from the earlier local
`results/current-model-diagnosis/` reconstruction, verified against its saved
manifest/checkpoint hashes. Baseline confusion counts match the original
record; exact identity with the original fitted-model/prediction artifact is
unavailable. Supplied outcomes reuse the earlier accepted-orders reconstruction
without fresh database reconciliation. No labels are constructed or queried.

The prediction stage verifies model SHA, runtime, feature order, parameters,
unlabeled inputs, and pre-snapshot training-label availability, then saves a
prediction checkpoint. A separate approved stage verifies that checkpoint
before reading explicit supplied outcomes. Neither stage fits a model or
optimizes a cutoff. The frozen bundle's `test_evaluated: false` remains its
immutable pre-evaluation state; the new evaluation manifest records this run.

The following commands document the completed authorization. Use a new directory
for any explicitly authorized rerun; read saved reports instead of rerunning.

```powershell
.venv/Scripts/python.exe comparisons/evaluate_frozen_october.py --stage predict --model results/repeated-grouped-precision --features results/current-model-diagnosis/unlabeled_october_scores.csv --feature-manifest results/current-model-diagnosis/manifest.json --output results/frozen-candidate-october-evaluation-NEW
.venv/Scripts/python.exe comparisons/evaluate_frozen_october.py --stage evaluate --output results/frozen-candidate-october-evaluation-NEW --outcomes results/current-model-diagnosis/october_cases.csv --outcome-manifest results/current-model-diagnosis/manifest.json --approve-inspected-evaluation
.venv/Scripts/python.exe comparisons/verify_frozen_october.py results/frozen-candidate-october-evaluation-NEW --model results/repeated-grouped-precision --features results/current-model-diagnosis/unlabeled_october_scores.csv
```

The completed run is `results/frozen-candidate-october-evaluation/`, retaining
protocol, prediction checkpoint, predictions, outcomes joined to scores, six
subgroup metric rows, paired decisions/changes, manifests, report, and independent
verification. Generated artifacts stay ignored. Seven synthetic tests cover
input/population/hash guards, explicit approval, checkpoint integrity, and a
no-fit two-stage smoke; they establish execution correctness, not model quality.
Independent verification exactly reproduces saved scores and all six metric
rows. The research bundle and operational model remain unchanged. No push,
merge, or deployment occurs. Further selection belongs on development data;
this inspected cohort cannot independently validate a newly selected method.

The subsequent [failure diagnosis](frozen_failure_diagnosis.md) audits the
cutoff-policy comparison, explains the precision change using subgroup counts
and operating rates, and examines stability using existing grouped development
and ablation predictions. It fits no models and changes no decision rule.

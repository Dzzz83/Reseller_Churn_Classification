# Authorized October test — latest paid-unit-revenue candidate

Frozen 2026-10-17 before final fitting or scoring. User explicitly requested:
“use that candidate and evaluate on the test set.” October is the test cohort;
no test outcome may select features, parameters, seeds, weights or cutoff.

Use the saved cadence_unit_revenue development specification unchanged: ten
original features + overdue_ratio + gap_cv + last_order_unit_revenue, seven-leaf
HGB, L2=1, other original parameters, balanced training weights, native missingness,
five-seed mean (42,78,88,1034,2026). Train on all 1,375 approved development rows.
Select its cutoff from three reseller-grouped OOF folds, seed78, maximizing
precision subject to recall strictly >90%, ties recall then higher cutoff.
Fit preprocessing/weights/models only within the corresponding training subset.
Do not target October's historical recall when selecting a cutoff.

Report three predetermined strategies on exactly the same492 eligible October
accounts: latest candidate, original ten-feature baseline trained with the same
OOF cutoff procedure, and saved historical baseline scores at0.06. The historical
rule remains the operational reference; matched cutoff policy isolates the
comparison under the development research objective. Neither reference is chosen
from test results. Reconstruct candidate features from accepted pre-October
purchases using existing history formulas, reconciling all13 development columns
and the ten saved October columns. No CompanyX ActualCost/current-profile fields.

Save protocol, training identities, OOF predictions/fold membership, frozen
cutoffs, full fitted models and492 unlabeled predictions per strategy before
reading supplied October labels. Separate evaluation verifies these hashes,
joins exact keys, and never fits or selects anything. Preserve all old outputs.

Primary operational check against EACH reference: candidate recall >= reference
and false positives < reference; report >=5% false-alert reduction separately.
These are point comparisons, not retrospective changes to the stricter failed
development acceptance. Report recall, precision, confusion counts, AP and AUC,
activity subgroups, all paired added/removed alerts and caught/missed churners.
Uncertainty: paired reseller bootstrap5000 draws seed2026, percentile95% intervals
for recall difference and false-alert reduction; resample all strategies together.
October has one row per reseller. Report empty-churn bootstrap draws if any.
Intervals condition on fitted models and omit training, cutoff, source and repeated
research uncertainty. No candidate/cutoff revision after evaluation, no deployment.

October outcomes have been inspected previously. This evaluates the chosen method
on the user's test set; it does not assert a new untouched independent cohort.
Saved baseline is a documented reproduction, not the unavailable original bundle.
Accepted-source completeness and order-time availability remain assumptions;
supplied labels were previously reconstructed from accepted clean orders.

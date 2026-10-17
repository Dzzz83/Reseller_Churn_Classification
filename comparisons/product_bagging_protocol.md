# Fixed reseller-bootstrap product ensemble — frozen before fitting

2026-10-17. The unbagged SKU candidate preserves recall in3 repeats, but FP
changes are -8,+45,+14 (reduction positive), mean17 (3.28%), below5%; both
conditional uncertainty checks fail. Its within-fold high-recall ranking improves,
but this alone is not operational success. Extended-history overdue differences
reverse, and only9 cases gain observable cadence; no history extension is selected.
Exact complete histories have only one mixed-outcome pair, insufficient to blame
most errors on exact input identity. Uniqueness is not proof of learnability.

All300 SKU and600 cadence study model identities have seed-score range0. The
five-seed mean therefore supplies no training-sample diversity. Repeat decision
variation persists; in the SKU active group actual disagreement8.52%, score-only
counterfactual8.60%, cutoff-only2.95%. These quantities are nonadditive and change
comparison pools; they do NOT isolate instability or prove bagging will help.
Hypothesis: a fixed reseller-bootstrap ensemble can reduce harmful training-
sample sensitivity without changing the historical information or cutoff objective.

ONE candidate: five bootstrap versions of the exact fixed cadence+SKU HGB.
For each estimator draw N reseller IDs with replacement from the sorted distinct
training reseller IDs, where N is their training count, using its original model
seed. Include every training snapshot of each drawn reseller once per draw.
This is training-only cluster resampling; natural inner/outer validation cases
are never sampled, duplicated or filtered. Require both training classes; record
and report failure, no resampling-until-favorable. Fit balanced class weights on
the resampled training rows. Each replica learns its unsupervised vocabulary
from DISTINCT sampled training resellers (max20 SKUs with support>=20); repeat
counts cannot inflate vocabulary support. Native missing values,7 leaves,L2=1
and all other parameters remain unchanged. Average the5 replica probabilities.
No bag-count/sample-fraction/seed/weight grid, replacement-feature search,
calibration or outer-outcome cutoff selection.

Use exact301/919/1201 outer partitions and cached3-fold inner membership. Original
balanced HGB remains PRIMARY comparator with unchanged saved scores/cutoffs.
Unbagged SKU and cadence are SECONDARY references only. Cutoff objective unchanged:
maximize inner-OOF precision with recall strictly>90%, ties recall then higher
cutoff. Fit300 new estimators; save resampled store sequences, training hashes,
vocabularies, feature order, estimator identities and scores before assessment.

Acceptance remains recall>=original baseline AND FP<baseline in EACH repeat,
>=5% mean FP reduction, <=5pp activity/snapshot mean recall loss. Require paired
5000-draw reseller-cluster recall lower>=0 and FP-reduction lower>0 at tail.05/48,
seed2026, preserving reseller snapshots/repetitions together. If ALL primary checks
pass, require same signs in50000 draws same seed/tail before acceptance. All
margins are research defaults. Report every fold/subgroup/tradeoff and conditional
limitations: bootstrap intervals omit model refitting/reselection and the full
adaptive project history. New training bootstraps are not independent validation
cohorts. No October outcome, new protected evaluation, promotion or deployment.
Failure does not authorize tuning bagging after seeing results.

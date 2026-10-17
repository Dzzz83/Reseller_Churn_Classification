# Historical product-mix trial — frozen before fitting

2026-10-16. The fixed-pair SKU screen used all132 historical products. Median
pair total variation is near1, showing that baseline-near cases can buy different
SKUs; this alone is not separation evidence. Two latest-order SKU shares have
same-direction one-point contrasts across all4 snapshots for persistent recent
false alerts. Controls/repeated snapshots are dependent and outcomes guided this
family investigation. Specific IDs are NOT selected for prediction from those
contrasts. Other groups/directions and all unflagged products remain reported.

Hypothesis: event product identities contain separating information beyond broad
category shares and diversity counts. Annual/latest counts already failed;
category and own-history overlap were previously tried. Archive ActualCost,
current product attributes, availability/end dates and reseller profiles remain
excluded. Accepted ProductID and LineTotal share the existing order-time-recording
assumption; the backup has no verified version history. No stronger availability
claim follows from this test.

One candidate: the fixed cadence-no-count HGB (original10 + overdue_ratio +
gap_cv,7 leaves,L2=1), augmented with annual AND latest-order SKU revenue shares.
Within EACH training partition choose at most20 products bought by at least20
DISTINCT training resellers in their annual histories. Rank by distinct reseller
count descending; ties numeric ProductID ascending. This unsupervised vocabulary
is fit separately within inner training and outer training. Include one annual
and one latest share per chosen SKU plus one other-share per scope. All unseen
or unselected IDs map to other. No ID/outcome association, target encoding,
validation support or global outcome screen selects vocabulary. No grid over
support, vocabulary length, windows or models. Sparse cases remain in evaluation.
The20-account support and20-product cap are research defaults to limit sparse
partitions, not business requirements or tuned values. Shares use [t-12m,t),
normalize by accepted line revenue and reconcile to SubTotal.

Primary comparator remains original balanced HGB10. Cadence-no-count is a
labelled secondary reference for attribution, not a weaker baseline. Reuse both
unchanged cached procedures on fixed301/919/1201 reseller-grouped outer splits
and matching3-fold inner membership,5 original model seeds, balanced TRAINING
weights, native missing values. Fit300 new estimators. Parameters and cutoff
objective unchanged: maximize inner-OOF precision with recall strictly>90%,
tie recall then higher cutoff. No outer-label cutoff, calibration or resampling.

Acceptance unchanged: recall>=original baseline and FP<baseline EACH repeat,
>=5% mean FP reduction, <=5pp mean recall loss by activity and snapshot. Report
all folds/repeats/subgroups/ranking near required recall, and failures. Paired
5000 reseller-cluster draws seed2026 carry all snapshots/repeats together;
recall lower>=0 and FP-reduction lower>0 at tail0.05/(23*2). If every primary
check passes, confirm same signs with50000 draws same seed/tail before acceptance.
These are research defaults. The conservative tail does not erase adaptive
reuse, training variability or future population uncertainty. Label results
exploratory development findings; no October outcomes, independent evaluation,
model promotion or deployment. Failure does not authorize tuning the vocabulary.

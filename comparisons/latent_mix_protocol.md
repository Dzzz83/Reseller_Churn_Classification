# Fixed training-only purchase-mix projection — frozen before fitting

2026-10-17. The five-replica reseller bootstrap fails full acceptance: mean5
fewer FP (0.97%), recall preserved but not fewer FP each repeat, uncertainty
fails. Do not retune seeds/bag count/sample fraction. Saved-vocabulary audit finds
that the fixed20 SKU inputs represent only21.00% of annual revenue and24.54% of
latest-order revenue on average; medians4.78%/3.37%. Most purchase value maps to
one other bin. FP and TP coverage both vary; outcome associations are descriptive,
not proof that uncovered products caused errors. Counts across repeats are dependent.

Hypothesis: a low-dimensional unsupervised representation of ALL training-observed
SKUs can retain useful purchase-mix structure lost by the single other bin while
using fewer predictor dimensions than direct capped SKU shares. This is a new
representation, not a cap/support grid or outcome-selected product list. It is
adaptively motivated by inspected development evidence, and remains exploratory.
Principal components need not have business/causal meaning. Information discarded
by dimensionality reduction remains a limitation; report explained variance.

ONE candidate: fixed cadence HGB12 (original10+overdue+gapCV,7 leaves,L2=1), plus
5 annual and5 latest-order projected revenue-share coordinates, plus unseen-SKU
share for each scope. Vocabulary is sorted ProductIDs with positive annual share
in that training partition only; no minimum support/cap or outcome association.
Fit a centered, unscaled PCA with5 components/full deterministic SVD on TRAINING
annual share rows only (snapshots are cases, no StoreID/snapshot predictor).
Apply the SAME learned mean/components to annual and latest profiles; keep
validation-only product mass in explicit unseen share. Each inner training and
outer training learns its own vocabulary/PCA. No global vocabulary, validation
fit, label-based projection, component-count/normalization grid or bagging.
Natural data have more than5 rows/SKUs; fail explicitly otherwise. Preserve row-
local half-open historical windows and accepted order-time-recording assumptions.
No current product attributes/archive cost/status changes.

Use exact301/919/1201 outer splits and saved3-fold inner membership;5 original
HGB seeds and balanced TRAINING weights/native missing values.300 new fits.
Primary comparator ORIGINAL balanced HGB10 remains cached and unchanged; cadence
and direct-SKU procedures are secondary references only. All share the inner
objective: maximize precision with recall strictly>90%, tie recall then higher
cutoff. No outer cutoff/calibration, protected outcomes or deployment.

Acceptance unchanged: recall>=original baseline and FP<baseline EACH repeat,
>=5% mean FP reduction, <=5pp activity/snapshot mean recall loss. Require paired
5000-reseller-cluster-draw lower recall>=0 and FP-reduction>0 at tail.05/(25*2),
seed2026, keeping snapshots/repeats together. If ALL primary checks pass, confirm
same signs with50000 draws same seed/tail. Record vocab, training means/components,
singular values/explained variance and expanded training-content hashes. Report
all outcomes/subgroups/failures and conditional uncertainty omissions: retraining,
selection, full adaptive history and future population shift. Research defaults
are not business requirements. Failure does not authorize component-count tuning.

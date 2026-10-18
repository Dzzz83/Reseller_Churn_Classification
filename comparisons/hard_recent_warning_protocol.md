# Missed recent-purchase development churners: warning-sign audit

User asks whether hard resellers show unused warning signs. This is a bounded
DEVELOPMENT-only descriptive investigation, after October was already inspected.
No October rows, labels, features, case histories or errors are inputs here.
No model fitting, cutoff tuning, label changes or automatic feature selection.
Existing October comparisons are not fresh confirmation evidence.

Fixed target: period/class-weighted model at its selected40%development precision
policy, using saved full-development grouped OOF predictions. Recently purchasing
means n_orders_3m>0. Targets are missed recent churners; reference groups are
caught recent churners and recent continuing purchasers. Keep every target.

Use the saved pre-snapshot [t-12m,t) histories and accepted-order trajectories.
Verify keys,counts,dates and baseline features against canonical development;
independently reconstruct the five interpretable historical ratios below.
Current ten-feature schema omits explicit quarter-to-quarter changes, personalized
cadence and last-order relative size, although existing recency/revenue/count
features can encode correlated information. No claim the model sees no warning.

Prespecified descriptive flags (undefined stays unknown, never assumed false):
1 recent3m revenue / previous3m revenue <1;
2 recent3m orders / previous3m orders <1;
3 last-order subtotal / mean earlier-order subtotal <1;
4 recency / mean historical interpurchase gap >1;
5 last historical interpurchase gap / mean earlier gaps >1.
These are possible warnings, not validated churn rules or causal indicators.
Report each flag,defined denominators and any-flag counts for all recent groups,
and each snapshot. Compare all18 previously recorded history descriptors, without
searching thresholds/directions for other quantities.

Matched comparisons: missed recent churners vs ALL recent non-churners (primary),
vs correctly unflagged recent non-churners (sensitivity), and vs caught recent
churners (context). Match within snapshot,on original10features only. Reuse the
existing equal-weight,log1p count/revenue,5th–95th span,clipped-distance procedure;
missing/missing0,missing/observed1. Exclude same reseller;StoreID breaks ties.
Report every target,coverage,distances,control reuse and boundary ties for k1/3/5.
No caliper or difficult-case removal. Match scales are descriptive,not fitting.

Report raw and paired feature/flag contrasts,means,medians,direction fractions,
missingness and allperiods/neighborhoods. Reused controls,repeated reseller rows,
selection on errors and many descriptors preclude an independent significance
claim. Do not call a feature useful merely because missed churners exhibit it;
non-churners can exhibit the same warning. Review earlier cadence/trend/basket
experiments before proposing a future development-only model experiment.

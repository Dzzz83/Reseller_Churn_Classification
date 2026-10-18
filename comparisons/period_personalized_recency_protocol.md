# Contingent personalized-recency comparison — frozen before fitting, 2026-10-18

Execute only if BOTH fixed product-engagement arms fail complete acceptance.
Preserve their decisions. This is a separate evidence-backed trial,not a new arm
retroactively appended to their protocol. No October inputs or evaluation.

Observed DEVELOPMENT evidence:binary 'overdue>1' flags do not distinguish the
missed recent-purchase cases,but that does not establish absence of conditional
information in the continuous ratio. Earlier controlled component ablation
f01 vs f00 changed ONLY overdue_ratio under ordinary balanced HGB. Mean within-fold
recent AP was .317870 vs .296230;differences positive in all three saved repeats
(.033995,.015510,.015415). Those historical operating comparisons did not establish
an accepted method; do not change their decisions. That controlled evidence is
stronger than comparing the multi-change cadence model with originalHGB.

Test ONE candidate: period_personalized_recency = original10 + overdue_ratio,
where overdue_ratio=recency_days / mean gap between DISTINCT accepted historical
purchase dates in[t-12m,t). Undefined short histories remain NaN. A ratio below1
can still carry conditional information; no threshold/direction is imposed.
No annual order count,gapCV,other cadence variable,regularization or feature grid.
Fitting weights remain fixed period/class N/(8*n_snapshot,class); original HGB
31leaves,L2=0,100iterations,learning_rate.1,minleaf20,early_stoppingFalse,
native missingness and originalfive seeds unchanged. This isolates ONE predictor
under the current fitting target,a meaningful difference from old class-weighted
ablation. No claimed novelty,causal mechanism or guaranteed information gain.

Use exactly period_product_ranking_protocol.md's approved cases,nested grouped
splits,period/class10 cached comparator,40%precision cutoff objective,tie rules,
fold/repeat/snapshot/activity/intersection reporting,ranking point gates,practical
and subgroup limits,5kbootstrap,conditional50k/exact confirmation,source/model/
weight identities and independent no-fit verification. No criterion weakens.
This is continuation cycle13:spend.001/2**12,tail spend/4=6.103515625e-8 across
two operating endpoints/both tails. Ranking95% intervals remain descriptive.
Prior adaptive development and inspected October do not become independent data.

Success still requires recentAPgain>=.01 and nonnegative eachrepeat,nonnegative
recentAUCgain,TPpreserved and fewerFP everyrepeat,>=5%meanFP reduction,<=5point
recallloss andzeroFPaddition eachperiod/activity/intersection,operating uncertainty.
Infeasible/missing/undefined gates block acceptance. Report failure honestly;
no lower-tail rescue,post-result tuning or automatic final test. If it fails,
review diagnostics and the need for genuinely different historical information
rather than cycling through previously exhausted models indefinitely.

Method context:the project historically evaluated behavioral recency/cadence;
BG/NBD motivates relating recency to purchasing frequency,not a theorem for this
six-month label: https://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf .
No NP/conformal/causal or model-quality guarantee is claimed.

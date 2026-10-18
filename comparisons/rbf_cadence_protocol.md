# Smooth multivariate margin learner — frozen 2026-10-18, contingent cycle 7

Execute ONLY if period-weighting cycle6 fails complete acceptance. Specific
hypothesis:the failed straight-line10feature and smooth additive12feature models
cannot represent nonlinear interactions,while tree-based procedures still have
period ranking losses. Test one smooth MULTIVARIATE RBF boundary on established
cadence12,not another knot/tree/cutoff grid. No prior SVC protocol was located in
reviewed comparison/config/model sources;this is not a novelty claim.

Training-only SimpleImputer(strategy=median,add_indicator=True,
keep_empty_features=True),then StandardScaler(),then SVC(C=1,kernel=rbf,
gamma=scale,class_weight=None,probability=False,tol=.001,max_iter=-1,
shrinking=True,cache_size=200). Supply original balanced FITTING sample weights
to SVC only;imputation/scaling/gamma use natural fitting rows. Inputs original10+
overdue_ratio+gap_cv. No logs,feature search,C/gamma grid,probability calibration
or resampling. Deterministic margin fit once perpartition:60fits. Record full
parameters,support indices/vectors/dual coefficients/intercept,effective gamma,
preprocessing statistics,training hashes/weights and convergence status.

Score=expit(decision_function),a strictly increasing bounded MARGIN SCORE,NOT
calibrated churn probability. No libsvm internal probabilityCV,which would not
preserve reseller groups by default. Ties remain intact. Same high-recall cutoff
objective for both baseline and candidate:precision-max pooledinnerOOF recall>.90,
tiesrecallthenhighercutoff,>=alerts. Outer labels select nothing.

Primary originalHGB10/cached baseline,identical1375rows/212churners/422resellers,
four approved snapshots;301/919/1201 x5outer,3innerseed78. All reseller snapshots
stay together inside both nested levels. Natural evaluation populations unchanged;
no source/history/data-role extension. Mixed-period grouped estimand only.

Complete acceptance UNCHANGED:recall>=baseline andFP<baseline EACH repeat;
>=5%meanFP reduction;each repeat's snapshot/activity/intersection <=.05recallloss,
ZEROaddedFP;undefined/infeasible/missing folds block. Paired5000reseller-cluster
drawsseed2026;cycle7spend.000015625 across2metrics/both tails -> .00000390625.
Require recalllower>=0,FPreductionlower>0;50000same-seed/tail confirmation ONLY
ifALLprimarychecks pass. Veryextreme empirical tails are imprecise and conditional
intervals exclude adaptive hypothesis choice,retraining/reselection and calendar
change. No moredraw rescue,parameter tuning or lowered requirements.

Correctness checks cover training-only transforms,missing indicators,constant and
extrapolated data,exact kernel/margin reconstruction and weighted-C dual bounds.
Independent no-fit audit reconstructs all60support/preprocessing/training
identities,held scores,30cuts,540metric rows,bootstrap/complete gates. Report
all folds/repeats/periods/groups/intersections,undefined flags,ranking/tied
outcome-derived points,cutoff transfer and support sizes. No final evaluation,
full-data model,promotion,push,merge or operational changes.

Primary method docs:https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
and https://scikit-learn.org/stable/modules/svm.html . RBF models require scale
handling and C/gamma choices;fixed defaults here are an explicit small research
comparison,not a claim of globally tuned superiority. Outlier-sensitive scaling
and imputation are limitations. Its smooth interactions differ from additive
splines;neither architecture alone guarantees improved recall/false alerts.

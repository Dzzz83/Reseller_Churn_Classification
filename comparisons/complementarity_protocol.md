# Baseline/cadence complementarity — exploratory protocol

Frozen 2026-10-16 after the user instructed continued work without further questions
and confirmed CompanyX.bak is the only source. Archive cost/version semantics remain
unresolved; no such field will be modeled. This cycle uses only established inputs.

Observed problem: cadence without annual count preserves recall and reduces FP
in all three new grouped repeats but misses its frozen practical/uncertainty rule.
Hypothesis to investigate before choosing a remedy: baseline and cadence might
retain different correct decisions and rankings near high recall, allowing a
simple fixed ensemble to exploit complementary errors. Five-seed averaging was
previously used; a fixed baseline/cadence blend has not been evaluated here.

Use only results/cadence-stability-20261016, with verified current-runtime scores
and inner OOF scores for seeds301/919/1201. Do not add seeds to the completed
cadence study or rewrite its failed decision. No model fitting, new cutoff
selection or October access in this exploratory step.

Report per repeat and activity: both caught, baseline-only caught,
cadence-only caught, both missed; both false alerts, baseline-only false alerts,
cadence-only false alerts, both correctly unalerted. Also report paired full-pool
and within-fold rank correlation, score scale/cutoff differences, within-fold
high-recall rank burden and disagreements among persistent original errors.
Distinguish decision complementarity from improved ranking; neither by itself
establishes benefit from an ensemble. Do not choose weights or a remedy from
October or from favorable individual folds. If a blend is motivated, a separate
protocol must freeze its weights, cutoff policy and complete acceptance rule
before reading any blended operating outcomes. Reuse of inspected outer cases
will be explicit; these are exploratory development results, not confirmation.

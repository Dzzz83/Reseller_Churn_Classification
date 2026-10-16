# Engineering instructions

Build and maintain the fixed ten-feature reseller churn pipeline. Experimental
searches and historical diagnostics remain on `pipeline_preparation`.

- Work on `final-pipeline-preparation` for this preparation; do not merge or push
  automatically. Preserve unrelated changes and tracked dataset references.
- Keep final implementation in `reseller_churn/`; `pipeline/` contains thin
  entry points. Commands, input contracts, and final results belong in README.md
  and docs/. Do not import research scripts or read dated experiment paths.
- Preserve historical information boundaries, training-only class weights,
  native missing discount values, and the fixed ensemble configuration.
- Predict without accessing outcomes. Cutoff evaluation uses explicit saved
  predictions and supplied outcomes; it never trains or optimizes cutoffs.
- October 2013 outcomes are already inspected. Further comparisons are
  descriptive. Fresh protected evaluation needs frozen methodology and explicit
  user approval; no automatic protected label construction is supported.
- Validate source coverage declarations and population joins. Record input,
  code, and model identities. Keep generated results/models and secrets ignored.
- Test with `python -m pytest -q`, then `python pipeline/smoke_test.py` when
  orchestration changes. Review the diff, report checks and limitations, and
  never present smoke results as model-quality evidence.

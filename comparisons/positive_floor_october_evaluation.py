"""No-fit outcome-stage recovery for string/date key normalization.

The original frozen runner reached the supplied-outcome join but failed on
mixed datetime/string snapshot types before calculating metrics. Preserve that
preparation and its source. This entry point only normalizes the join key;
models, scores, cutoffs, labels and metric definitions remain unchanged.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.evaluate_frozen_october import matched_outcomes, verified_artifact
from comparisons.positive_floor_october import KEYS, ROOT, checked_stage, paired_changes
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    write_json,
    write_manifest,
)


def join_supplied_outcomes(predictions, labels, expected_rows=492):
    normalized = predictions.copy()
    normalized["snapshot"] = pd.to_datetime(normalized.snapshot).dt.strftime("%Y-%m-%d")
    return matched_outcomes(normalized, labels, expected_rows=expected_rows)


def evaluate(output):
    checked_stage(output)
    target = output / "completed_evaluation"
    require(not target.exists(), "Preserve previous evaluation")
    outcomes = ROOT / "results/current-model-diagnosis/october_cases.csv"
    verified_artifact(outcomes.parent / "manifest.json", outcomes)
    # First outcome read occurs only after the prediction checkpoint is verified.
    labels = pd.read_csv(outcomes, usecols=KEYS + ["churn"])
    predictions = read_csv(output / "unlabeled_predictions.csv")
    require(
        predictions.predicted_churn.eq(
            predictions.score.ge(predictions.cutoff).astype(int)
        ).all(),
        "Changed decisions",
    )
    cases = join_supplied_outcomes(predictions, labels)
    out = new_output_directory(target)
    cases.to_csv(out / "evaluated_cases.csv", index=False)
    rows = []
    for strategy, part in cases.groupby("strategy"):
        for name, subset in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            rows.append(
                dict(
                    strategy=strategy,
                    subgroup=name,
                    resellers=subset.StoreID.nunique(),
                    **study.decision_metrics(subset.churn, subset.score, subset.cutoff),
                    recall_defined=bool(subset.churn.sum() > 0),
                    precision_defined=bool(subset.predicted_churn.sum() > 0),
                    roc_auc=float(roc_auc_score(subset.churn, subset.score))
                    if subset.churn.nunique() == 2
                    else None,
                )
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    changes = paired_changes(cases)
    write_json(out / "paired_changes.json", changes)
    report = "# Authorized inspected October evaluation — positive-floor veto\n\n"
    report += (
        "Fixed development procedure; October previously inspected, so this is descriptive reuse, not independent testing. No test tuning. Development acceptance remains FAILED.\n\n```text\n"
        + metrics.to_string(index=False)
        + "\n```\n\n"
    )
    report += (
        "Paired change versus the matched-cutoff baseline, with descriptive conditional 95% reseller-bootstrap intervals:\n\n```json\n"
        + json.dumps(changes, indent=2)
        + "\n```\n\n"
    )
    report += "Predictions checkpointed before supplied outcomes were joined. Historical 0.06 is a separate policy reference. No labels constructed, source queries, cutoff revisions or deployment. Source coverage and historical availability remain assumptions; prior inspected outcomes are supplied reconstructed labels. Conditional bootstrap omits refitting, cutoff/floor selection, adaptive research and future population shift.\n"
    (out / "report.md").write_text(report)
    write_manifest(
        out,
        dict(
            stage="evaluated",
            models_fitted=0,
            cutoffs_selected=0,
            inputs=input_identity(
                {
                    "prediction_checkpoint": output / "manifest.json",
                    "supplied_outcomes": outcomes,
                    "outcome_entrypoint": Path(__file__),
                }
            ),
        ),
    )
    print(metrics[metrics.subgroup.eq("overall")].to_string(index=False))
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    evaluate(parser.parse_args().directory)

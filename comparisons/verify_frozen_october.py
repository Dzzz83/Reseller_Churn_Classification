"""Verify checkpointed frozen-candidate evaluation without fitting any model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from comparisons import evaluate_frozen_october as final
from comparisons import grouped_precision_recall as study


def audit(directory, model_directory, features_path):
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or study.file_hash(path) != expected
        ):
            raise ValueError(f"Evaluation checksum mismatch: {name}")
    protocol = json.loads((directory / "protocol.json").read_text())
    if (
        manifest["models_fitted"] != 0
        or manifest["thresholds_optimized"]
        or manifest["independent_evaluation"]
    ):
        raise ValueError("Evaluation did not follow the frozen descriptive protocol")
    if (
        study.file_hash(features_path) != protocol["features_sha256"]
        or study.file_hash(model_directory / "research_candidate.joblib")
        != protocol["model_sha256"]
    ):
        raise ValueError("Scoring input/model identity changed")
    features = study.read_csv(features_path).sort_values("StoreID")
    models, specification, _ = final.load_frozen_model(model_directory)
    predictions = study.read_csv(directory / "unlabeled_predictions.csv")
    for strategy, expected, cutoff in (
        ("original_baseline", features.score.to_numpy(), 0.06),
        ("frozen_cadence_candidate", None, specification["cutoff"]),
    ):
        part = predictions[predictions.strategy.eq(strategy)].sort_values("StoreID")
        np.testing.assert_array_equal(part.StoreID, features.StoreID)
        if expected is None:
            with threadpool_limits(limits=1):
                expected = np.mean(
                    [
                        model.predict_proba(
                            features[specification["candidate"]["features"]]
                        )[:, 1]
                        for model in models
                    ],
                    axis=0,
                )
        np.testing.assert_array_equal(part.score, expected)
        if (
            not part.cutoff.eq(cutoff).all()
            or not part.predicted_churn.eq(part.score.ge(cutoff).astype(int)).all()
        ):
            raise ValueError("Decisions differ from frozen cutoffs")
    cases = study.read_csv(directory / "evaluated_cases.csv")
    if cases.duplicated(["strategy", *study.KEYS]).any():
        raise ValueError("Duplicate evaluated keys")
    if (
        cases.groupby("strategy").size().ne(492).any()
        or cases.groupby("strategy").churn.sum().ne(46).any()
    ):
        raise ValueError("Unexpected cohort counts")
    metrics = study.read_csv(directory / "metrics.csv")
    for row in metrics.itertuples():
        part = cases[cases.strategy.eq(row.strategy)]
        if row.subgroup == "active":
            part = part[part.n_orders_3m > 0]
        if row.subgroup == "inactive":
            part = part[part.n_orders_3m == 0]
        y = part.churn.eq(1).to_numpy()
        alert = part.score.ge(part.cutoff).to_numpy()
        tp, fp, fn, tn = [
            int(value.sum())
            for value in (y & alert, ~y & alert, y & ~alert, ~y & ~alert)
        ]
        if (row.tp, row.fp, row.fn, row.tn) != (tp, fp, fn, tn):
            raise ValueError("Confusion counts differ")
        np.testing.assert_allclose(
            [row.precision, row.recall, row.f1, row.average_precision, row.roc_auc],
            [
                tp / max(1, tp + fp),
                tp / max(1, tp + fn),
                2 * tp / max(1, 2 * tp + fp + fn),
                average_precision_score(y, part.score),
                roc_auc_score(y, part.score),
            ],
            atol=1e-15,
            rtol=0,
        )
    overall = metrics[metrics.subgroup.eq("overall")].set_index("strategy")
    old = overall.loc["original_baseline"]
    new = overall.loc["frozen_cadence_candidate"]
    if (old.tp, old.fp, old.fn) != (43, 108, 3):
        raise ValueError("Reproduced baseline differs from historical counts")
    table = [
        "| Model / frozen cutoff | Recall | Precision | Caught | Missed | False alerts | Alerts | AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, row in overall.iterrows():
        table.append(
            f"| {strategy} / {row.cutoff:.10f} | {row.recall:.2%} | {row.precision:.2%} | {int(row.tp)} | {int(row.fn)} | {int(row.fp)} | {int(row.alerts)} | {row.average_precision:.4f} |"
        )
    changed = json.loads((directory / "paired_changes.json").read_text())
    subgroups = [
        "| Model | Subgroup | Recall | Precision | Caught | False alerts |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics[metrics.subgroup.ne("overall")].itertuples():
        subgroups.append(
            f"| {row.strategy} | {row.subgroup} | {row.recall:.2%} | {row.precision:.2%} | {row.tp} | {row.fp} |"
        )
    report = f"""# Frozen candidate evaluation on the inspected October cohort

The frozen candidate **fails the requested precision-improvement objective**
on this cohort: recall is {new.recall:.2%}, but precision is {new.precision:.2%}
versus {old.precision:.2%} for the original rule. It catches
{int(new.tp - old.tp)} additional churners with {int(new.fp - old.fp)} net additional
false alerts. No cutoff or parameter was changed after evaluation.

## Frozen method and cohort

User explicitly requested: "evaluate using the final test set." The evaluated
candidate was saved before this request: original ten features plus overdue
ratio, annual order count, and purchase-gap variation, seven leaves, L2=1,
five registered model seeds averaged, native missing handling, training-only
balanced weights, and development-selected cutoff `{specification["cutoff"]}`.
The original comparison uses its preserved 0.06 rule. No alternative feature
set or cutoff was evaluated to choose a test winner.

Scoring date is 2013-10-01; outcomes cover [2013-10-01,2014-04-01).
There are 492 accounts and 46 churners. October outcomes were already
inspected before this follow-up. This is **descriptive reuse**, not a new
independent test. It does not retroactively change the original test record.

The supplied unlabeled scoring CSV and supplied outcome CSV have separate
checksums from the earlier reconstruction. Prediction loads the existing
frozen bundle, verifies parameters/runtime/input identities, and writes a
checkpoint. A separate explicitly approved evaluation stage verifies that
checkpoint before reading supplied labels. Neither stage fits a model,
optimizes cutoffs, queries a database, or constructs outcomes.

## Results

{chr(10).join(table)}

Precision difference: {(new.precision - old.precision) * 100:+.2f} percentage points.
Recall difference: {(new.recall - old.recall) * 100:+.2f} points.
Paired decisions: {changed["newly_caught"]} newly caught churners,
{changed["newly_missed"]} newly missed, {changed["false_alerts_removed"]} false
alerts removed, and {changed["false_alerts_added"]} added.

{chr(10).join(subgroups)}

The net false-alert increase occurs among active accounts. Overall ranking AP
improves, but active-subgroup AP falls from 0.1553 to 0.0765; inactive-subgroup
AP improves from 0.8870 to 0.9469. The aggregate ranking improvement therefore
does not establish better active-account discrimination or a better frozen
operating point. Scores are not validated calibrated individual risks.

Grouped development gains did not transfer into improved October precision
at the frozen cutoff. Do not promote this candidate as meeting the user's
objective on this cohort. Retain the result and, if further development is
authorized, choose changes using development data rather than optimizing this
inspected test result.

## Provenance and verification

All model, feature, prediction-checkpoint, and supplied-outcome hashes are
recorded. Independent checks reproduce every score from the saved model,
all six metric rows, alert equality at the frozen cutoff, and the original
baseline confusion counts. Source model and cutoff remain unchanged.

Original baseline scores are a documented local reproduction; exact original
artifact identity is unavailable. Counts match the recorded 43 caught, three
missed, and 108 false alerts. Outcomes reuse the earlier accepted-orders CSV
reconstruction; no fresh source-database reconciliation is performed here.
These source limitations and prior cohort inspection remain material.

The seven new synthetic tests cover unlabeled scoring, scope/population/hash
guards, explicit approval, checkpoint integrity, and a no-fit two-stage smoke.
Synthetic smoke checks execution only. The broader repository suite remains
blocked by the previously documented config module/package collision.

No training, test-driven retuning, operational-model replacement, push, or
merge was performed.
"""
    (directory / "report.md").write_text(report, encoding="utf-8")
    verification = dict(
        status="passed",
        score_parity_exact=True,
        metric_rows_verified=len(metrics),
        model_file_unchanged=True,
        cutoff_unchanged=True,
        models_fitted=0,
        thresholds_optimized=False,
        independent_evaluation=False,
    )
    study.write_json(directory / "verification.json", verification)
    study.write_json(
        directory / "report_manifest.json",
        dict(
            verifier_sha256=study.file_hash(__file__),
            evaluation_manifest_sha256=study.file_hash(
                directory / "evaluation_manifest.json"
            ),
            outputs={
                name: study.file_hash(directory / name)
                for name in ("report.md", "verification.json")
            },
        ),
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.directory, arguments.model, arguments.features)

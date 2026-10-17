"""No-fit verification and reporting recovery for the authorized October run.

Preserves the checkpoint and partial evaluation, including the exact entrypoint
as run. Serialization can change joblib estimator hashes; verify predictions,
parameters, training identities and the immutable serialized bundle instead.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from comparisons import evaluate_unit_revenue_october as final
from comparisons import grouped_precision_recall as study
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)


def verify(output):
    verify_manifest(output)
    protocol = json.loads((output / "protocol.json").read_text())
    archive = output / "verification/evaluation_entrypoint_as_run.py"
    for name, identity in protocol["inputs"].items():
        path = archive if name == "entrypoint" else Path(identity["path"])
        assert study.file_hash(path) == identity["sha256"], name
    data = study.read_csv(protocol["inputs"]["development"]["path"])
    data.snapshot = pd.to_datetime(data.snapshot)
    features = study.read_csv(output / "unlabeled_features.csv")
    predictions = study.read_csv(output / "unlabeled_predictions.csv")
    bundles = joblib.load(output / "frozen_models.joblib")
    oof = study.read_csv(output / "development_oof.csv")
    identities = json.loads((output / "training_identities.json").read_text())
    splits = json.loads((output / "training_splits.json").read_text())["folds"]
    for fold, split in enumerate(splits):
        ti, vi = split["train"], split["validation"]
        assert not set(data.iloc[ti].StoreID) & set(data.iloc[vi].StoreID)
        for spec in protocol["arms"]:
            rows = data.iloc[ti][study.KEYS + ["churn"] + spec["features"]]
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(rows, index=False).values.tobytes()
            ).hexdigest()
            fitted = [
                item
                for item in identities["fits"]
                if item["candidate"] == spec["id"]
                and item["stage"] == "inner"
                and item["inner_fold"] == fold
            ]
            assert len(fitted) == 5 and all(
                item["training_sha256"] == digest for item in fitted
            )
            saved = oof[oof.candidate.eq(spec["id"]) & oof.inner_fold.eq(fold)]
            assert set(saved.StoreID) == set(data.iloc[vi].StoreID)
    hash_changes = 0
    for name, bundle in bundles.items():
        spec = bundle["specification"]
        assert spec == next(item for item in protocol["arms"] if item["id"] == name)
        for seed, model in zip(study.MODEL_SEEDS, bundle["models"]):
            assert list(model.feature_names_in_) == spec["features"]
            assert all(
                model.get_params()[key] == value
                for key, value in {**spec["parameters"], "random_state": seed}.items()
            )
        part = predictions[predictions.strategy.eq(name)]
        scores = np.mean(
            [
                model.predict_proba(features[spec["features"]])[:, 1]
                for model in bundle["models"]
            ],
            axis=0,
        )
        np.testing.assert_allclose(scores, part.score, rtol=0, atol=1e-15)
        assert features.StoreID.tolist() == part.StoreID.tolist()
        inner = oof[oof.candidate.eq(name)]
        assert len(inner) == 1375 and not inner.duplicated(study.KEYS).any()
        assert (
            study.select_threshold(inner.churn, inner.score)["cutoff"]
            == bundle["cutoff"]
        )
        assert part.cutoff.eq(bundle["cutoff"]).all()
        rows = data[study.KEYS + ["churn"] + spec["features"]]
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(rows, index=False).values.tobytes()
        ).hexdigest()
        full = [
            item
            for item in identities["fits"]
            if item["stage"] == "full" and item["candidate"] == name
        ]
        assert len(full) == 5 and all(
            item["training_sha256"] == digest for item in full
        )
        hash_changes += sum(
            item["estimator_hash"] != joblib.hash(model)
            for item, model in zip(full, bundle["models"])
        )
    assert len(identities["fits"]) == 40
    assert predictions.predicted_churn.eq(
        predictions.score.ge(predictions.cutoff).astype(int)
    ).all()
    historical = (
        study.read_csv(final.SAVED / "unlabeled_october_scores.csv")
        .set_index(study.KEYS)
        .sort_index()
    )
    saved = (
        predictions[predictions.strategy.eq("historical_0.06")]
        .set_index(study.KEYS)
        .sort_index()
    )
    np.testing.assert_array_equal(historical.score, saved.score)
    assert saved.cutoff.eq(0.06).all()
    labels_path = final.SAVED / "october_cases.csv"
    final.verified_artifact(final.SAVED / "manifest.json", labels_path)
    cases = final.matched_outcomes(
        predictions, pd.read_csv(labels_path, usecols=study.KEYS + ["churn"])
    )
    stored = study.read_csv(output / "evaluation/evaluated_cases.csv")
    pd.testing.assert_frame_equal(cases, stored)
    metrics = study.read_csv(output / "evaluation/metrics.csv")
    for row in metrics.itertuples(index=False):
        part = cases[cases.strategy.eq(row.strategy)]
        if row.subgroup == "recent_purchase":
            part = part[part.n_orders_3m.gt(0)]
        elif row.subgroup == "no_recent_purchase":
            part = part[part.n_orders_3m.eq(0)]
        tn, fp, fn, tp = confusion_matrix(
            part.churn, part.predicted_churn, labels=[0, 1]
        ).ravel()
        assert (tn, fp, fn, tp) == (row.tn, row.fp, row.fn, row.tp)
        assert row.rows == len(part) and row.churners == part.churn.sum()
        assert np.isclose(row.recall, tp / (tp + fn)) and np.isclose(
            row.precision, tp / (tp + fp)
        )
    changes = json.loads((output / "evaluation/paired_changes.json").read_text())
    assert changes["comparisons"] == [
        final.paired_summary(cases, name) for name in ("historical_0.06", "baseline")
    ]
    out = new_output_directory(output / "completed_report")
    overall = metrics[metrics.subgroup.eq("overall")]
    report = "# Authorized October test evaluation\n\n"
    report += "The candidate fails the false-alert reduction objective against both predeclared references. No test-driven tuning followed.\n\n"
    report += "```text\n" + overall.to_string(index=False) + "\n```\n\n"
    report += "The candidate catches45/46 (97.83%) with194 false alerts and18.83% precision. Historical0.06 catches43/46 (93.48%) with108 false alerts and28.48% precision. The matched development-only cutoff baseline catches46/46 with183 false alerts and20.09% precision.\n\n"
    report += "Against historical0.06: two additional churners caught,86 additional false alerts (97 added,11 removed). Against the matched-policy baseline: one fewer churner caught and11 additional false alerts (48 added,37 removed). All additional false alerts and the candidate miss are in the recently purchasing subgroup. All35 inactive churners and10 inactive continuing buyers are alerted by every strategy.\n\n"
    report += "Paired reseller bootstrap95% intervals (5000 draws,seed2026): historical comparison recall difference[0,+11.36]percentage points, false-alert reduction[-105,-67]; matched-policy comparison recall difference[-7.32,0]points, false-alert reduction[-29,+7]. No empty-churn draws. Intervals condition on fitted procedures and omit training/cutoff/source/research uncertainty.\n\n"
    report += "The higher overall AP(0.7585 vs0.7013) does not establish operational improvement; recent-purchase AP is lower(0.0695 vs0.1553). No replacement or deployment is justified. Full subgroup metrics and paired changes remain in ../evaluation/.\n\n"
    report += "Predictions/cutoffs/models were saved before this evaluation read supplied labels. October was inspected historically; this is the user-authorized October test, not a newly untouched cohort. Accepted-source completeness/order-time availability remain assumptions; supplied labels were reconstructed previously. The historical baseline is a documented reproduction. Stores reference bytes differ only in LF/CRLF; historical CRLF SHA matches exactly.\n\n"
    report += "Execution notes: the original evaluation saved all cases,metrics and paired intervals, then its optional pandas Markdown renderer failed because tabulate was absent. This no-fit verifier independently reproduces saved-model scores,cutoffs,40 training hashes,group isolation and confusion counts, and completes this report without changing those outputs. The exact entrypoint as run is archived under ../verification/. In-memory estimator joblib hashes change after serialization for all10 models; file SHA and reloaded prediction/parameter parity pass. The reporting entrypoint now uses plain text tables to remove that optional dependency; no methodological code changed.\n"
    (out / "report.md").write_text(report)
    write_json(
        out / "verification.json",
        dict(
            fits_verified=40,
            serialized_models_verified=10,
            cutoffs_reproduced=2,
            confusion_rows_verified=9,
            test_cases_per_strategy=492,
            no_refits=True,
            estimator_hash_changes_after_serialization=hash_changes,
            source_archive_sha256=study.file_hash(archive),
            paired_intervals_reproduced=True,
        ),
    )
    write_manifest(
        out,
        dict(
            inputs=input_identity(
                {
                    "prediction_manifest": output / "manifest.json",
                    "cases": output / "evaluation/evaluated_cases.csv",
                    "metrics": output / "evaluation/metrics.csv",
                    "changes": output / "evaluation/paired_changes.json",
                    "outcomes": labels_path,
                    "entrypoint_as_run": archive,
                    "verifier": Path(__file__),
                }
            )
        ),
    )
    print(overall.to_string(index=False))
    print(
        "All saved-model,training,cutoff,decision,confusion and paired-bootstrap checks passed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)

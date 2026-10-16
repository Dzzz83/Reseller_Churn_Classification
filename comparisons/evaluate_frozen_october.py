"""Score the frozen candidate, then evaluate supplied inspected October outcomes.

Two explicit stages preserve a prediction checkpoint. Neither stage trains,
selects a model, optimizes a cutoff, queries a database, or constructs labels.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study

SNAPSHOT = "2013-10-01"
LABEL_END = "2014-04-01"


def verified_artifact(manifest_path, path):
    manifest = json.loads(Path(manifest_path).read_text())
    relative = str(
        Path(path).resolve().relative_to(Path(manifest_path).resolve().parent)
    )
    expected = manifest["outputs"].get(relative)
    if expected is None or study.file_hash(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {relative}")
    return manifest


def load_frozen_model(directory):
    directory = Path(directory)
    manifest = directory / "manifest.json"
    for name in (
        "frozen_candidate.json",
        "research_candidate.joblib",
        "protocol.json",
        "runtime_identity.json",
    ):
        verified_artifact(manifest, directory / name)
    specification = json.loads((directory / "frozen_candidate.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    runtime = json.loads((directory / "runtime_identity.json").read_text())
    current = dict(
        python=platform.python_version(),
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        joblib=joblib.__version__,
    )
    if any(runtime[name] != value for name, value in current.items()):
        raise ValueError("Saved model runtime differs from the current environment")
    if (
        specification["protocol_sha256"] != study.file_hash(directory / "protocol.json")
        or protocol["test_data_used"]
    ):
        raise ValueError("Model is not the frozen development-only selection")
    if (
        specification["candidate"]["id"] != "cadence_regularized"
        or specification["candidate"]["features"] != study.FEATURES + study.CADENCE
    ):
        raise ValueError("Unexpected frozen feature set")
    if max(
        pd.Timestamp(date) + pd.DateOffset(months=6) for date in protocol["snapshots"]
    ) > pd.Timestamp(SNAPSHOT):
        raise ValueError("Training labels mature after the test snapshot")
    bundle = joblib.load(directory / "research_candidate.joblib")
    for name in ("candidate", "cutoff", "training_sha256", "protocol_sha256"):
        if bundle["specification"][name] != specification[name]:
            raise ValueError("Bundle and frozen specification disagree")
    if len(bundle["models"]) != len(specification["seeds"]):
        raise ValueError("Wrong ensemble size")
    for model, seed in zip(bundle["models"], specification["seeds"]):
        if (
            not isinstance(model, HistGradientBoostingClassifier)
            or list(model.feature_names_in_) != specification["candidate"]["features"]
        ):
            raise ValueError("Wrong estimator or feature order")
        for name, value in {
            **specification["candidate"]["parameters"],
            "random_state": seed,
        }.items():
            if model.get_params()[name] != value:
                raise ValueError("Frozen parameters differ")
    return bundle["models"], specification, current


def validate_features(frame, columns, expected_rows=492):
    if {"churn", "actual_churn", "future_orders", "result"} & set(frame):
        raise ValueError("Scoring input must be unlabeled")
    required = [*study.KEYS, "score", *columns]
    if set(required) - set(frame):
        raise ValueError("Scoring columns missing")
    if (
        len(frame) != expected_rows
        or frame[study.KEYS].isna().any().any()
        or frame.duplicated(study.KEYS).any()
    ):
        raise ValueError("Wrong or duplicate scoring population")
    if not pd.to_datetime(frame.snapshot).eq(pd.Timestamp(SNAPSHOT)).all():
        raise ValueError(
            "Only the explicitly authorized inspected October snapshot is supported"
        )
    if (
        not frame.StoreID.gt(0).all()
        or not frame.StoreID.eq(np.floor(frame.StoreID)).all()
    ):
        raise ValueError("Invalid reseller keys")
    values = frame[columns].to_numpy(dtype=float)
    if (
        np.isinf(values).any()
        or not np.isfinite(frame[study.FEATURES[:7]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Invalid scoring features")
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Invalid saved baseline scores")
    return frame.sort_values("StoreID").reset_index(drop=True)


def predict(model_directory, features_path, feature_manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve previous evaluation; choose a new directory")
    verified_artifact(feature_manifest, features_path)
    source_checkpoint = Path(feature_manifest).parent / "prediction_checkpoint.json"
    verified_artifact(feature_manifest, source_checkpoint)
    baseline = json.loads(source_checkpoint.read_text())
    if baseline["score_sha256"] != study.file_hash(features_path):
        raise ValueError("Baseline feature/score checkpoint differs")
    if (
        baseline["cutoff"] != 0.06
        or baseline["features"] != study.FEATURES
        or baseline["model"] != study.BASE_PARAMETERS
        or tuple(baseline["seeds"]) != study.MODEL_SEEDS
    ):
        raise ValueError(
            "Saved baseline differs from the documented original methodology"
        )
    models, specification, runtime = load_frozen_model(model_directory)
    columns = specification["candidate"]["features"]
    frame = validate_features(study.read_csv(features_path), columns)
    output.mkdir(parents=True)
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization="User: evaluate using the final test set.",
        cohort_status="Previously inspected October cohort; descriptive reuse",
        snapshot=SNAPSHOT,
        outcome_window_end_exclusive=LABEL_END,
        model_specification=specification,
        baseline_specification=baseline,
        script_sha256=study.file_hash(__file__),
        runtime=runtime,
        model_sha256=study.file_hash(
            Path(model_directory) / "research_candidate.joblib"
        ),
        model_manifest_sha256=study.file_hash(Path(model_directory) / "manifest.json"),
        features_sha256=study.file_hash(features_path),
        feature_manifest_sha256=study.file_hash(feature_manifest),
        models_fitted=0,
        thresholds_optimized=False,
        independent_evaluation=False,
        limitations=[
            "October outcomes were inspected before this follow-up; not a new independent test",
            "Saved supplied outcomes were reconstructed earlier from accepted clean orders without fresh database reconciliation",
            "Original baseline scores are a documented reproduction; exact original artifact identity is unavailable",
            "Evaluate only the previously frozen candidate/cutoff; no test-driven retuning",
        ],
    )
    study.write_json(output / "protocol.json", protocol)
    with threadpool_limits(limits=1):
        scores = np.mean(
            [model.predict_proba(frame[columns])[:, 1] for model in models], axis=0
        )
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid candidate scores")
    predictions = []
    for strategy, values, cutoff in (
        ("original_baseline", frame.score.to_numpy(), 0.06),
        ("frozen_cadence_candidate", scores, specification["cutoff"]),
    ):
        part = frame[study.KEYS + ["n_orders_3m"]].copy()
        part["strategy"], part["score"], part["cutoff"] = strategy, values, cutoff
        part["predicted_churn"] = part.score.ge(cutoff).astype(int)
        part = part.sort_values(["score", "StoreID"], ascending=[False, True])
        part["rank"] = np.arange(1, len(part) + 1)
        predictions.append(part)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / "unlabeled_predictions.csv", index=False
    )
    study.write_json(
        output / "prediction_checkpoint.json",
        dict(
            saved_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=study.file_hash(output / "protocol.json"),
            predictions_sha256=study.file_hash(output / "unlabeled_predictions.csv"),
            outcomes_read_by_this_stage=False,
            models_fitted=0,
            rows_per_strategy=len(frame),
        ),
    )
    print(
        "Frozen candidate predictions checkpointed; no outcomes read and no models fitted."
    )


def matched_outcomes(predictions, outcomes, expected_rows=492):
    labels = outcomes.copy()
    if set([*study.KEYS, "churn"]) - set(labels):
        raise ValueError("Outcome columns missing")
    labels = labels[[*study.KEYS, "churn"]]
    labels["snapshot"] = pd.to_datetime(labels.snapshot).dt.strftime("%Y-%m-%d")
    if (
        labels.duplicated(study.KEYS).any()
        or labels[study.KEYS].isna().any().any()
        or not labels.churn.isin([0, 1]).all()
    ):
        raise ValueError("Outcomes must have unique keys and complete binary labels")
    if len(labels) != expected_rows or not labels.snapshot.eq(SNAPSHOT).all():
        raise ValueError("Wrong inspected outcome population")
    result = []
    for _, part in predictions.groupby("strategy"):
        merged = part.merge(
            labels, on=study.KEYS, how="outer", validate="one_to_one", indicator=True
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError("Prediction/outcome populations differ")
        result.append(merged.drop(columns="_merge"))
    return pd.concat(result, ignore_index=True)


def evaluate(output, outcomes_path, outcome_manifest, approve_inspected=False):
    if not approve_inspected:
        raise ValueError(
            "Explicit approval to evaluate the inspected cohort is required"
        )
    output = Path(output)
    if (output / "evaluation_manifest.json").exists():
        raise FileExistsError("Evaluation already complete; preserve it")
    protocol = json.loads((output / "protocol.json").read_text())
    checkpoint = json.loads((output / "prediction_checkpoint.json").read_text())
    if (
        study.file_hash(output / "protocol.json") != checkpoint["protocol_sha256"]
        or study.file_hash(output / "unlabeled_predictions.csv")
        != checkpoint["predictions_sha256"]
    ):
        raise ValueError("Prediction checkpoint changed")
    verified_artifact(outcome_manifest, outcomes_path)
    predictions = study.read_csv(output / "unlabeled_predictions.csv")
    # This explicit evaluation stage reads supplied, previously inspected labels.
    outcomes = pd.read_csv(outcomes_path, usecols=[*study.KEYS, "churn"])
    cases = matched_outcomes(predictions, outcomes)
    if cases.groupby("strategy").churn.sum().ne(46).any():
        raise ValueError("Unexpected inspected churn population")
    if not cases.predicted_churn.eq(cases.score.ge(cases.cutoff).astype(int)).all():
        raise ValueError("Checkpoint decisions differ from frozen cutoffs")
    rows = []
    for strategy, part in cases.groupby("strategy"):
        expected_cutoff = (
            0.06
            if strategy == "original_baseline"
            else protocol["model_specification"]["cutoff"]
        )
        if not part.cutoff.eq(expected_cutoff).all():
            raise ValueError("Frozen cutoff changed")
        for subgroup, subset in (
            ("overall", part),
            ("active", part[part.n_orders_3m.gt(0)]),
            ("inactive", part[part.n_orders_3m.eq(0)]),
        ):
            values = study.decision_metrics(subset.churn, subset.score, subset.cutoff)
            values["f1"] = (
                2
                * values["tp"]
                / max(1, 2 * values["tp"] + values["fp"] + values["fn"])
            )
            values["roc_auc"] = (
                float(roc_auc_score(subset.churn, subset.score))
                if subset.churn.nunique() == 2
                else None
            )
            rows.append(
                dict(
                    strategy=strategy,
                    subgroup=subgroup,
                    cutoff=expected_cutoff,
                    **values,
                )
            )
    metrics = pd.DataFrame(rows)
    cases.to_csv(output / "evaluated_cases.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    baseline = cases[cases.strategy.eq("original_baseline")]
    candidate = cases[cases.strategy.eq("frozen_cadence_candidate")]
    paired = baseline[study.KEYS + ["churn", "predicted_churn"]].merge(
        candidate[study.KEYS + ["predicted_churn"]],
        on=study.KEYS,
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    positive = paired.churn.eq(1)
    old, new = (
        paired.predicted_churn_baseline.eq(1),
        paired.predicted_churn_candidate.eq(1),
    )
    changed = dict(
        newly_caught=int((positive & ~old & new).sum()),
        newly_missed=int((positive & old & ~new).sum()),
        false_alerts_removed=int((~positive & old & ~new).sum()),
        false_alerts_added=int((~positive & ~old & new).sum()),
    )
    paired.to_csv(output / "paired_decisions.csv", index=False)
    study.write_json(output / "paired_changes.json", changed)
    study.write_json(
        output / "evaluation_manifest.json",
        dict(
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            outcomes_sha256=study.file_hash(outcomes_path),
            outcome_manifest_sha256=study.file_hash(outcome_manifest),
            outcomes_source=str(Path(outcomes_path).resolve()),
            checkpoint_sha256=study.file_hash(output / "prediction_checkpoint.json"),
            script_sha256=study.file_hash(__file__),
            models_fitted=0,
            thresholds_optimized=False,
            independent_evaluation=False,
            cohort_status="inspected",
            outputs={
                path.name: study.file_hash(path)
                for path in output.iterdir()
                if path.is_file()
            },
        ),
    )
    print(metrics[metrics.subgroup.eq("overall")].to_string(index=False))
    print(json.dumps(changed, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["predict", "evaluate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--outcome-manifest", type=Path)
    parser.add_argument("--approve-inspected-evaluation", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage == "predict":
        if (
            any(
                value is None
                for value in (
                    arguments.model,
                    arguments.features,
                    arguments.feature_manifest,
                )
            )
            or arguments.outcomes is not None
        ):
            parser.error(
                "Predict requires model/features/feature-manifest and accepts no outcomes"
            )
        predict(
            arguments.model,
            arguments.features,
            arguments.feature_manifest,
            arguments.output,
        )
    else:
        if arguments.outcomes is None or arguments.outcome_manifest is None:
            parser.error("Explicit supplied outcomes and their manifest are required")
        evaluate(
            arguments.output,
            arguments.outcomes,
            arguments.outcome_manifest,
            arguments.approve_inspected_evaluation,
        )

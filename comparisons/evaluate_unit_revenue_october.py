"""Freeze development-only fits/predictions, then evaluate authorized October labels."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.history_comparison import history_descriptors
from comparisons.baseline_diagnosis.inputs import require
from comparisons.evaluate_frozen_october import (
    SNAPSHOT,
    matched_outcomes,
    validate_features,
    verified_artifact,
)
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)
from reseller_churn.data.preparation import build_features, load_sources

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "results/cadence-unit-revenue-20261017"
SAVED = ROOT / "results/current-model-diagnosis"
PROTOCOL = ROOT / "comparisons/unit_revenue_october_protocol.md"
CANDIDATE = "cadence_unit_revenue"


def reconcile(left, right, columns):
    """Compare by identity, preserving missingness and allowing CSV rounding only."""
    require(not left.duplicated(study.KEYS).any(), "Duplicate reference keys")
    require(not right.duplicated(study.KEYS).any(), "Duplicate rebuilt keys")
    a = left.set_index(study.KEYS).sort_index()
    b = right.set_index(study.KEYS).sort_index()
    require(a.index.equals(b.index), "Feature populations differ")
    np.testing.assert_allclose(
        a[columns], b[columns], rtol=1e-10, atol=1e-9, equal_nan=True
    )


def feature_inputs(data):
    directory = ROOT / "inputs/regeneration-20261016"
    manifest_path = directory / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = dict(
        orders=ROOT
        / "results/research-provenance-20261016/orders_clean_historical_bytes.csv",
        stores=ROOT / "datasets/stores_clean.csv",
        headers=directory / "order_headers.csv",
        lines=directory / "order_lines.csv",
    )
    preparation = json.loads(
        (ROOT / "results/development-precision-diagnosis/manifest.json").read_text()
    )
    require(
        study.file_hash(manifest_path)
        == preparation["input_hashes"]["source_manifest"],
        "Changed source declaration",
    )
    for name in ("headers", "lines"):
        require(
            study.file_hash(paths[name])
            == manifest["exports"][paths[name].name]["sha256"],
            "Changed export",
        )
    for name in ("orders", "stores"):
        expected = manifest["reference_inputs"][f"datasets\\{name}_clean.csv"]
        raw = paths[name].read_bytes()
        historical_bytes = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        require(
            study.file_hash(paths[name]) == expected
            or hashlib.sha256(historical_bytes).hexdigest() == expected,
            "Changed clean source beyond line endings",
        )
    coverage = manifest["coverage"]
    source = load_sources(
        paths, coverage["from"], coverage["through_exclusive"], SNAPSHOT
    )
    additions = ["overdue_ratio", "gap_cv", "last_order_unit_revenue"]
    keys = pd.concat(
        [data[study.KEYS], build_features(source, SNAPSHOT)[study.KEYS]],
        ignore_index=True,
    )
    history, _ = history_descriptors(keys, source.orders, source.lines)
    rebuilt = pd.concat(
        [build_features(source, snapshot) for snapshot in keys.snapshot.unique()],
        ignore_index=True,
    )
    rebuilt = rebuilt.merge(
        history[study.KEYS + additions], on=study.KEYS, validate="one_to_one"
    )
    reconcile(
        data,
        rebuilt[rebuilt.snapshot.lt(pd.Timestamp(SNAPSHOT))],
        study.FEATURES + additions,
    )
    verified_artifact(SAVED / "manifest.json", SAVED / "unlabeled_october_scores.csv")
    verified_artifact(SAVED / "manifest.json", SAVED / "prediction_checkpoint.json")
    saved = study.read_csv(SAVED / "unlabeled_october_scores.csv")
    baseline = json.loads((SAVED / "prediction_checkpoint.json").read_text())
    require(
        baseline["score_sha256"]
        == study.file_hash(SAVED / "unlabeled_october_scores.csv"),
        "Changed baseline checkpoint",
    )
    require(
        baseline["cutoff"] == 0.06
        and baseline["features"] == study.FEATURES
        and baseline["model"] == study.BASE_PARAMETERS
        and tuple(baseline["seeds"]) == study.MODEL_SEEDS,
        "Changed baseline policy",
    )
    saved.snapshot = pd.to_datetime(saved.snapshot)
    test = rebuilt[rebuilt.snapshot.eq(pd.Timestamp(SNAPSHOT))].copy()
    reconcile(saved, test, study.FEATURES)
    test = test.merge(
        saved[study.KEYS + ["score"]], on=study.KEYS, validate="one_to_one"
    )
    test = validate_features(test, study.FEATURES + additions)
    require(
        np.isfinite(test.last_order_unit_revenue).all()
        and test.last_order_unit_revenue.gt(0).all(),
        "Invalid unit revenue",
    )
    return test, {
        **paths,
        "source_manifest": manifest_path,
        "saved_baseline": SAVED / "unlabeled_october_scores.csv",
    }


def prepare(output):
    require(not output.exists(), "Preserve previous evaluation")
    verify_manifest(RESEARCH)
    data = study.read_csv(RESEARCH / "development_input.csv")
    data.snapshot = pd.to_datetime(data.snapshot)
    require(
        len(data) == 1375 and data.churn.sum() == 212 and data.StoreID.nunique() == 422,
        "Wrong development population",
    )
    require(
        set(data.snapshot.dt.strftime("%Y-%m-%d"))
        == {"2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01"},
        "Wrong training dates",
    )
    protocol = json.loads((RESEARCH / "protocol.json").read_text())
    grid = protocol["arms"]
    require([item["id"] for item in grid] == ["baseline", CANDIDATE], "Wrong candidate")
    require(
        grid[1]["features"]
        == study.FEATURES + ["overdue_ratio", "gap_cv", "last_order_unit_revenue"],
        "Wrong features",
    )
    require(
        grid[1]["parameters"]
        == {**study.BASE_PARAMETERS, "max_leaf_nodes": 7, "l2_regularization": 1.0},
        "Wrong parameters",
    )
    require(
        grid[0]["features"] == study.FEATURES
        and grid[0]["parameters"] == study.BASE_PARAMETERS,
        "Wrong baseline",
    )
    test, inputs = feature_inputs(data)
    out = new_output_directory(output)
    (out / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    write_json(
        out / "protocol.json",
        dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            arms=grid,
            seeds=list(study.MODEL_SEEDS),
            inner_folds=3,
            inner_seed=78,
            authorization="User: use that candidate and evaluate on the test set.",
            inputs=input_identity(
                {
                    **inputs,
                    "development": RESEARCH / "development_input.csv",
                    "research_protocol": RESEARCH / "protocol.json",
                    "entrypoint": Path(__file__),
                    "fitter": Path(study.__file__),
                    "history_builder": ROOT
                    / "comparisons/baseline_diagnosis/history_comparison.py",
                }
            ),
            bootstrap_draws=5000,
            bootstrap_seed=2026,
            interval=[0.025, 0.975],
            outcome_access=False,
            cutoff_objective="Maximize OOF precision, recall>0.90; ties recall then cutoff",
        ),
    )
    test.to_csv(out / "unlabeled_features.csv", index=False)
    # No October labels are accepted or read in this stage.
    selections, identities, oof = study.inner_selection(
        data, grid, 3, 78, study.MODEL_SEEDS
    )
    oof.to_csv(out / "development_oof.csv", index=False)
    splits = study.grouped_splits(data, 3, 78)
    write_json(
        out / "training_splits.json",
        {
            "folds": [
                dict(train=ti.tolist(), validation=vi.tolist()) for ti, vi in splits
            ]
        },
    )
    parts, bundles = [], {}
    for spec, selection in zip(grid, selections):
        scores, fitted, spread, models = study.fit_scores(
            data, test[spec["features"]], spec, retain_models=True
        )
        identities.extend(
            [{**item, "stage": "full", "seed_score_range": spread} for item in fitted]
        )
        bundles[spec["id"]] = dict(
            models=models, specification=spec, cutoff=selection["cutoff"]
        )
        part = test[study.KEYS + ["n_orders_3m"]].copy()
        part["strategy"], part["score"], part["cutoff"] = (
            spec["id"],
            scores,
            selection["cutoff"],
        )
        parts.append(part)
    historical = test[study.KEYS + ["n_orders_3m", "score"]].copy()
    historical["strategy"], historical["cutoff"] = "historical_0.06", 0.06
    parts.append(historical)
    predictions = pd.concat(parts, ignore_index=True)
    predictions["predicted_churn"] = predictions.score.ge(predictions.cutoff).astype(
        int
    )
    predictions.to_csv(out / "unlabeled_predictions.csv", index=False)
    joblib.dump(bundles, out / "frozen_models.joblib")
    write_json(
        out / "training_identities.json", {"fits": identities, "selections": selections}
    )
    write_manifest(
        out,
        dict(
            stage="predictions_checkpointed",
            outcomes_read=False,
            fitted_models=len(identities),
        ),
    )
    print(
        "Predictions and models frozen; no October labels read. Cutoffs:",
        selections,
        flush=True,
    )


def paired_summary(cases, reference, draws=5000, seed=2026):
    candidate = cases[cases.strategy.eq(CANDIDATE)].set_index(study.KEYS).sort_index()
    baseline = cases[cases.strategy.eq(reference)].set_index(study.KEYS).sort_index()
    require(candidate.index.equals(baseline.index), "Unmatched paired populations")
    require(candidate.churn.equals(baseline.churn), "Unmatched paired labels")
    y = candidate.churn.to_numpy().astype(bool)
    old = baseline.predicted_churn.to_numpy().astype(bool)
    new = candidate.predicted_churn.to_numpy().astype(bool)
    delta = new.astype(int) - old.astype(int)
    rng = np.random.default_rng(seed)
    recalls, false_alerts = [], []
    skipped = 0
    for _ in range(draws):
        sample = rng.integers(0, len(y), len(y))
        if not y[sample].any():
            skipped += 1
            continue
        recalls.append(float(delta[sample][y[sample]].mean()))
        false_alerts.append(int(-delta[sample][~y[sample]].sum()))
    return dict(
        reference=reference,
        newly_caught=int((y & ~old & new).sum()),
        newly_missed=int((y & old & ~new).sum()),
        false_alerts_removed=int((~y & old & ~new).sum()),
        false_alerts_added=int((~y & ~old & new).sum()),
        recall_difference=float(delta[y].mean()),
        false_alert_reduction=int(-delta[~y].sum()),
        point_objective_met=bool(delta[y].sum() >= 0 and delta[~y].sum() < 0),
        practical_5pct_met=bool(-delta[~y].sum() >= 0.05 * (old & ~y).sum()),
        recall_difference_interval=np.quantile(recalls, [0.025, 0.975]).tolist(),
        false_alert_reduction_interval=np.quantile(
            false_alerts, [0.025, 0.975]
        ).tolist(),
        empty_churn_draws=skipped,
    )


def checked_prediction_stage(output):
    manifest = verify_manifest(output)
    require(
        manifest["stage"] == "predictions_checkpointed"
        and not manifest["outcomes_read"],
        "Wrong prediction stage",
    )
    protocol = json.loads((output / "protocol.json").read_text())
    for identity in protocol["inputs"].values():
        require(
            study.file_hash(identity["path"]) == identity["sha256"],
            "Frozen input or code changed",
        )
    return protocol


def evaluate(output):
    protocol = checked_prediction_stage(output)
    out = new_output_directory(output / "evaluation")
    # Outcome reading is deliberately downstream of the verified saved predictions.
    outcomes_path = SAVED / "october_cases.csv"
    verified_artifact(SAVED / "manifest.json", outcomes_path)
    labels = pd.read_csv(outcomes_path, usecols=study.KEYS + ["churn"])
    predictions = study.read_csv(output / "unlabeled_predictions.csv")
    require(
        predictions.predicted_churn.eq(
            predictions.score.ge(predictions.cutoff).astype(int)
        ).all(),
        "Changed decisions",
    )
    cases = matched_outcomes(predictions, labels)
    rows = []
    for strategy, part in cases.groupby("strategy"):
        for group, subset in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            rows.append(
                dict(
                    strategy=strategy,
                    subgroup=group,
                    **study.decision_metrics(subset.churn, subset.score, subset.cutoff),
                    roc_auc=float(roc_auc_score(subset.churn, subset.score))
                    if subset.churn.nunique() == 2
                    else None,
                )
            )
    metrics = pd.DataFrame(rows)
    changes = [
        paired_summary(
            cases, reference, protocol["bootstrap_draws"], protocol["bootstrap_seed"]
        )
        for reference in ("historical_0.06", "baseline")
    ]
    cases.to_csv(out / "evaluated_cases.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    write_json(out / "paired_changes.json", {"comparisons": changes})
    report = "# October test evaluation — latest unit-revenue candidate\n\n"
    report += "Fixed development-only features, parameters, weights and cutoff procedure. No test tuning.\n\n"
    report += "```text\n" + metrics.to_string(index=False) + "\n```\n\n"
    report += (
        "Paired changes and percentile95% reseller bootstrap intervals:\n\n```json\n"
        + json.dumps(changes, indent=2)
        + "\n```\n\n"
    )
    report += "Both references are predeclared; historical0.06 is the operational rule. This test does not override the failed full development acceptance. October was inspected historically; source completeness/order-time availability remain assumptions, labels are supplied reconstructed outcomes, and the saved historical baseline is a reproduction. Bootstrap conditions on these fitted models and omits training/cutoff/source/research uncertainty. No deployment or test-driven revision.\n"
    (out / "report.md").write_text(report)
    write_manifest(
        out,
        dict(
            stage="evaluated",
            inputs=input_identity(
                {
                    "prediction_manifest": output / "manifest.json",
                    "outcomes": outcomes_path,
                }
            ),
            models_fitted=0,
            cutoffs_selected=0,
        ),
    )
    print(metrics[metrics.subgroup.eq("overall")].to_string(index=False))
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "evaluate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else evaluate)(args.output)

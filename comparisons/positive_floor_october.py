"""Two-stage authorized October evaluation of the fixed positive-floor veto."""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.crossfit_percentile import FEATURES, key_records
from comparisons.development_checks import validate_development_cases
from comparisons.evaluate_frozen_october import (
    matched_outcomes,
    validate_features,
    verified_artifact,
)
from comparisons.positive_floor_veto import apply_veto, learn_veto
from comparisons.retained_calibration import calibrate, score_logits
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "positive_floor_veto"
KEYS = study.KEYS
PREVIOUS = ROOT / "results/unit-revenue-october-evaluation-20261017"
DEVELOPMENT = ROOT / "results/retained-calibration-20261018"
PROTOCOL = ROOT / "comparisons/positive_floor_october_protocol.md"


def fit_retained(train, held):
    """Keep each classifier with the map learned from disjoint reference resellers."""
    spec = dict(
        id="crossfit_components",
        features=FEATURES,
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    values, records, retained = [], [], []
    for half, (ti, ri) in enumerate(study.grouped_splits(train, 2, 42)):
        fitting, reference = train.iloc[ti], train.iloc[ri]
        require(set(fitting.StoreID).isdisjoint(reference.StoreID), "Reference leakage")
        _, identities, _, models = study.fit_scores(
            fitting, held[FEATURES], spec, retain_models=True
        )
        for model, identity in zip(models, identities):
            ref = model.predict_proba(reference[FEATURES])[:, 1]
            raw = model.predict_proba(held[FEATURES])[:, 1]
            mapped, mapping = calibrate(ref, reference.churn, raw)
            values.append(mapped)
            retained.append(
                dict(
                    model=model, slope=mapping["slope"], intercept=mapping["intercept"]
                )
            )
            records.append(
                {
                    **identity,
                    **mapping,
                    "half": half,
                    "reference_scores": ref.tolist(),
                    "reference_cases": key_records(reference),
                    "fitting_cases": key_records(fitting),
                    "held_cases": key_records(held),
                    "raw_held_scores": raw.tolist(),
                }
            )
    return np.mean(values, axis=0), records, retained


def predict_retained(models, features):
    return np.mean(
        [
            expit(
                pair["slope"]
                * score_logits(
                    pair["model"].predict_proba(features[FEATURES])[:, 1]
                ).ravel()
                + pair["intercept"]
            )
            for pair in models
        ],
        axis=0,
    )


def prepare(output):
    require(not output.exists(), "Preserve previous evaluation")
    verify_manifest(DEVELOPMENT)
    verify_manifest(PREVIOUS)
    data = read_csv(DEVELOPMENT / "development_input.csv")
    validate_development_cases(data, FEATURES)
    pd.testing.assert_frame_equal(
        data,
        read_csv(ROOT / "results/cadence-unit-revenue-20261017/development_input.csv")[
            data.columns
        ],
    )
    require(
        max(data.snapshot + pd.DateOffset(months=6)) <= pd.Timestamp("2013-10-01"),
        "Unmatured training labels",
    )
    previous_protocol = json.loads((PREVIOUS / "protocol.json").read_text())
    baseline_spec = previous_protocol["arms"][0]
    require(
        baseline_spec["features"] == study.FEATURES
        and baseline_spec["parameters"] == study.BASE_PARAMETERS,
        "Cached baseline specification changed",
    )
    require(
        tuple(previous_protocol["seeds"]) == study.MODEL_SEEDS
        and previous_protocol["inner_folds"] == 3
        and previous_protocol["inner_seed"] == 78,
        "Cached fitting procedure changed",
    )
    # Scoring input is the previously checkpointed UNLABELED table only.
    test = validate_features(read_csv(PREVIOUS / "unlabeled_features.csv"), FEATURES)
    out = new_output_directory(output)
    shutil.copyfile(PROTOCOL, out / "protocol.md")
    paths = {
        "runner": Path(__file__),
        "veto": ROOT / "comparisons/positive_floor_veto.py",
        "calibration": ROOT / "comparisons/retained_calibration.py",
        "fitter": Path(study.__file__),
    }
    for name, path in paths.items():
        shutil.copyfile(path, out / ("as_run_" + name + ".py"))
    write_json(
        out / "protocol.json",
        dict(
            authorization="User: evaluate on the final test set.",
            independent_test=False,
            development_acceptance=False,
            features=FEATURES,
            seeds=list(study.MODEL_SEEDS),
            baseline_spec=baseline_spec,
            component_parameters={
                **study.BASE_PARAMETERS,
                "max_leaf_nodes": 7,
                "l2_regularization": 1.0,
            },
            calibrator=dict(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-8),
            inputs=input_identity(
                {
                    **paths,
                    "development": DEVELOPMENT / "development_input.csv",
                    "previous_manifest": PREVIOUS / "manifest.json",
                    "frozen_protocol": PROTOCOL,
                }
            ),
            runtime=source_identity()["versions"],
            outcomes_read=False,
        ),
    )
    data.to_csv(out / "development_input.csv", index=False)
    test.to_csv(out / "unlabeled_features.csv", index=False)
    baseline = read_csv(PREVIOUS / "development_oof.csv")
    baseline = baseline[baseline.candidate.eq("baseline")].copy()
    component_parts, identities, split_records = [], [], []
    for fold, (ti, vi) in enumerate(study.grouped_splits(data, 3, 78)):
        fitting, held = data.iloc[ti], data.iloc[vi]
        require(
            set(fitting.StoreID).isdisjoint(held.StoreID), "Inner held reseller leakage"
        )
        expected = (
            baseline[baseline.inner_fold.eq(fold)]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            expected[KEYS + ["churn"]],
            held[KEYS + ["churn"]].sort_values(KEYS).reset_index(drop=True),
        )
        scores, fitted, _ = fit_retained(fitting, held[KEYS + FEATURES])
        identities.extend([{**r, "stage": "inner", "inner_fold": fold} for r in fitted])
        component_parts.append(
            held[KEYS + ["churn"]].assign(
                score=scores, candidate="retained_logistic", inner_fold=fold
            )
        )
        split_records.append(dict(fold=fold, train=ti.tolist(), validation=vi.tolist()))
        print(f"Inner fold {fold} complete", flush=True)
    component = pd.concat(component_parts, ignore_index=True)
    selection, transformed, protected = learn_veto(baseline, component)
    write_json(out / "selection.json", selection)
    protected.to_csv(out / "protected_inner_cases.csv", index=False)
    pd.concat(
        [baseline, component, transformed.assign(candidate=CANDIDATE)],
        ignore_index=True,
    ).to_csv(out / "development_oof.csv", index=False)
    write_json(out / "training_splits.json", dict(folds=split_records))
    # The full-development rule is frozen above before scoring October.
    scores, fitted, retained = fit_retained(data, test[KEYS + FEATURES])
    identities.extend([{**r, "stage": "full"} for r in fitted])
    old_bundle = joblib.load(PREVIOUS / "frozen_models.joblib")["baseline"]
    require(
        old_bundle["specification"] == baseline_spec
        and old_bundle["cutoff"] == selection["reference_cutoff"],
        "Baseline full rule differs",
    )
    require(
        len(old_bundle["models"]) == len(study.MODEL_SEEDS),
        "Baseline ensemble size differs",
    )
    for model, seed in zip(old_bundle["models"], study.MODEL_SEEDS):
        require(
            list(model.feature_names_in_) == study.FEATURES, "Baseline schema differs"
        )
        require(
            all(
                model.get_params()[name] == value
                for name, value in {
                    **study.BASE_PARAMETERS,
                    "random_state": seed,
                }.items()
            ),
            "Baseline parameters differ",
        )
    base_scores = np.mean(
        [
            model.predict_proba(test[study.FEATURES])[:, 1]
            for model in old_bundle["models"]
        ],
        axis=0,
    )
    previous_predictions = read_csv(PREVIOUS / "unlabeled_predictions.csv")
    cached = (
        previous_predictions[previous_predictions.strategy.eq("baseline")]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    np.testing.assert_array_equal(base_scores, cached.score)
    base = test[KEYS + ["n_orders_3m"]].assign(
        score=base_scores,
        cutoff=selection["reference_cutoff"],
        strategy="baseline",
        churn=0,
    )
    # Dummy equal labels are used only for identity alignment, then immediately dropped.
    extra = base.assign(score=scores)
    veto = apply_veto(base, extra, selection).assign(strategy=CANDIDATE)
    historical = test[KEYS + ["n_orders_3m"]].assign(
        score=test.score, cutoff=0.06, strategy="historical_0.06", churn=0
    )
    predictions = pd.concat([base, veto, historical], ignore_index=True).drop(
        columns="churn"
    )
    predictions["predicted_churn"] = predictions.score.ge(predictions.cutoff).astype(
        int
    )
    predictions.to_csv(out / "unlabeled_predictions.csv", index=False)
    test[KEYS].assign(score=scores).to_csv(out / "component_scores.csv", index=False)
    joblib.dump(
        dict(baseline=old_bundle["models"], component=retained, selection=selection),
        out / "frozen_models.joblib",
    )
    write_json(
        out / "training_identities.json",
        dict(
            fits=identities,
            new_hgb_fits=40,
            new_logistic_fits=40,
            cached_baseline_hgb=5,
        ),
    )
    write_manifest(out, dict(stage="predictions_checkpointed", outcomes_read=False))
    print("Predictions checkpointed before outcomes:", selection, flush=True)


def checked_stage(output):
    manifest = verify_manifest(output)
    require(
        manifest["stage"] == "predictions_checkpointed"
        and not manifest["outcomes_read"],
        "No verified prediction checkpoint",
    )
    protocol = json.loads((output / "protocol.json").read_text())
    for identity in protocol["inputs"].values():
        require(
            study.file_hash(identity["path"]) == identity["sha256"],
            "Frozen code/input changed",
        )
    require(protocol["runtime"] == source_identity()["versions"], "Runtime changed")
    return protocol


def paired_changes(cases):
    base = cases[cases.strategy.eq("baseline")].sort_values(KEYS).reset_index(drop=True)
    new = cases[cases.strategy.eq(CANDIDATE)].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(base[KEYS + ["churn"]], new[KEYS + ["churn"]])
    y = base.churn.to_numpy().astype(bool)
    a = base.predicted_churn.to_numpy().astype(bool)
    b = new.predicted_churn.to_numpy().astype(bool)
    require((~b | a).all(), "Veto added an alert")
    weights = np.random.default_rng(2026).multinomial(
        len(base), np.repeat(1 / len(base), len(base)), size=5000
    )
    counts = weights @ y
    valid = counts > 0
    recall = (weights @ ((b.astype(int) - a) * y))[valid] / counts[valid]
    fp = weights @ ((a.astype(int) - b) * (~y))
    return dict(
        newly_caught=int((y & ~a & b).sum()),
        newly_missed=int((y & a & ~b).sum()),
        false_alerts_removed=int((~y & a & ~b).sum()),
        false_alerts_added=int((~y & ~a & b).sum()),
        recall_difference_interval=np.quantile(recall, [0.025, 0.975]).tolist(),
        fp_reduction_interval=np.quantile(fp, [0.025, 0.975]).tolist(),
        undefined_draws=int((~valid).sum()),
        bootstrap_draws=5000,
        seed=2026,
    )


def evaluate(output):
    checked_stage(output)
    target = output / "evaluation"
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
    cases = matched_outcomes(predictions, labels)
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
                    rows=len(subset),
                    resellers=subset.StoreID.nunique(),
                    **study.decision_metrics(subset.churn, subset.score, subset.cutoff),
                    recall_defined=bool(subset.churn.sum() > 0),
                    precision_defined=bool(subset.predicted_churn.sum() > 0),
                    average_precision=float(
                        average_precision_score(subset.churn, subset.score)
                    ),
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
                }
            ),
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

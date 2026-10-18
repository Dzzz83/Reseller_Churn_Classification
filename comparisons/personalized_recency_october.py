"""User-authorized precision-target evaluation of fixed personalized recency."""

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.all_candidate_precision import ordered, score_models
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.development_checks import validate_development_cases
from comparisons.evaluate_frozen_october import verified_artifact
from comparisons.period_weighting import period_weights
from comparisons.positive_floor_october_evaluation import join_supplied_outcomes
from comparisons.precision_floor_october import (
    cutoff_scan,
    independent_choice,
    select_floor,
)
from comparisons.reseller_weighting import fit_weighted
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "results/all-candidate-precision-october-20261018-04"
RESEARCH = ROOT / "results/period-personalized-recency-20261018"
TARGETS = list(range(20, 71, 10))
NAME = "period_personalized_recency"


def prepare(output):
    out = new_output_directory(output)
    write_json(
        out / "protocol.json",
        dict(
            authorization="User requests final-test evaluation and addition to the 20–70% table.",
            candidate=NAME,
            features=study.FEATURES + ["overdue_ratio"],
            parameters=study.BASE_PARAMETERS,
            seeds=study.MODEL_SEEDS,
            weights="N/(8*n_snapshot,class), computed within each fitting partition",
            targets=TARGETS,
            folds=3,
            fold_seed=78,
            objective="Maximum pooled development OOF recall subject to precision >= target; ties higher precision then higher cutoff; intact score ties.",
            infeasible="No cutoff or test policy; undefined precision is not success.",
            scoring="Fit on all approved development after OOF; no test fitting or cutoff tuning.",
            status="Previously inspected October, descriptive reuse. Existing failed development acceptance unchanged.",
            input_identities=input_identity(
                dict(research=RESEARCH / "manifest.json", prior=PRIOR / "manifest.json")
            ),
            runtime=source_identity(),
        ),
    )
    sources = new_output_directory(out / "source_snapshot")
    for name in [
        "personalized_recency_october",
        "reseller_weighting",
        "period_weighting",
        "precision_floor_october",
        "all_candidate_precision",
        "grouped_precision_recall",
    ]:
        shutil.copyfile(ROOT / "comparisons" / f"{name}.py", sources / f"{name}.py")
    verify_manifest(RESEARCH)
    verify_manifest(PRIOR)
    data = ordered(read_csv(RESEARCH / "development_input.csv"))
    validate_development_cases(data)
    earlier = ordered(read_csv(PRIOR / "development_input.csv"))
    columns = study.KEYS + ["churn"] + study.FEATURES + ["overdue_ratio"]
    pd.testing.assert_frame_equal(data[columns], earlier[columns])
    test = ordered(read_csv(PRIOR / "unlabeled_features.csv"))
    require(
        "churn" not in test and len(test) == 492 and test.StoreID.nunique() == 492,
        "Scoring population differs",
    )
    require(
        test.snapshot.eq(pd.Timestamp("2013-10-01")).all(), "Scoring snapshot differs"
    )
    require(
        (data.snapshot + pd.DateOffset(months=6)).max() <= test.snapshot.min(),
        "Immature training labels",
    )
    spec = dict(
        id=NAME,
        features=study.FEATURES + ["overdue_ratio"],
        parameters=study.BASE_PARAMETERS,
    )
    data[columns].to_csv(out / "development_input.csv", index=False)
    test[study.KEYS + spec["features"]].to_csv(
        out / "unlabeled_features.csv", index=False
    )
    splits = study.grouped_splits(data, 3, 78)
    write_json(
        out / "splits.json",
        [dict(train=ti.tolist(), validation=vi.tolist()) for ti, vi in splits],
    )
    values = np.full(len(data), np.nan)
    records = []
    for fold, (ti, vi) in enumerate(splits):
        score, fitted, _, models = fit_weighted(
            data.iloc[ti],
            data.iloc[vi][study.KEYS + spec["features"]],
            spec,
            weight_function=period_weights,
            retain_models=True,
        )
        bundle = dict(spec=spec, models=models)
        path = out / f"inner_{fold}_models.joblib"
        joblib.dump(bundle, path)
        np.testing.assert_allclose(
            score, score_models(joblib.load(path), data.iloc[vi]), rtol=0, atol=1e-15
        )
        values[vi] = score
        records.extend([dict(stage="inner", fold=fold, **r) for r in fitted])
        print(f"Development fold {fold} completed and reload verified", flush=True)
    require(np.isfinite(values).all(), "Missing OOF scores")
    data[study.KEYS + ["churn"]].assign(score=values).to_csv(
        out / "development_oof.csv", index=False
    )
    scan = cutoff_scan(data.churn, values)
    scan.to_csv(out / "development_cutoff_scan.csv", index=False)
    choices = []
    for target in TARGETS:
        choice = select_floor(scan, target)
        direct = independent_choice(data.churn, values, target)
        require(
            (choice is None and direct is None)
            or (choice is not None and choice["cutoff"] == direct),
            "Independent cutoff selection differs",
        )
        choices.append(
            dict(
                candidate=NAME,
                minimum_precision_percent=target,
                feasible=choice is not None,
                **(choice or {}),
            )
        )
    pd.DataFrame(choices).to_csv(out / "development_selections.csv", index=False)
    scores, fitted, _, models = fit_weighted(
        data,
        test[study.KEYS + spec["features"]],
        spec,
        weight_function=period_weights,
        retain_models=True,
    )
    joblib.dump(dict(spec=spec, models=models), out / "frozen_models.joblib")
    np.testing.assert_allclose(
        scores,
        score_models(joblib.load(out / "frozen_models.joblib"), test),
        rtol=0,
        atol=1e-15,
    )
    records.extend([dict(stage="full", **r) for r in fitted])
    write_json(out / "training_identities.json", records)
    parts = []
    for choice in choices:
        if choice["feasible"]:
            part = test[study.KEYS + ["n_orders_3m"]].assign(
                candidate=NAME,
                strategy=f"{NAME}_{choice['minimum_precision_percent']}",
                minimum_precision_percent=choice["minimum_precision_percent"],
                score=scores,
                cutoff=choice["cutoff"],
            )
            part["predicted_churn"] = part.score.ge(part.cutoff).astype(int)
            parts.append(part)
    pd.concat(parts, ignore_index=True).to_csv(
        out / "unlabeled_predictions.csv", index=False
    )
    write_manifest(
        out,
        dict(stage="predictions_checkpointed", outcomes_read=False, new_model_fits=20),
    )
    verify_manifest(out)


def evaluate(output):
    require(
        verify_manifest(output)["stage"] == "predictions_checkpointed",
        "Wrong checkpoint",
    )
    out = new_output_directory(output / "evaluation")
    shutil.copyfile(Path(__file__), out / "as_run_evaluator.py")
    predictions = read_csv(output / "unlabeled_predictions.csv")
    require(
        predictions.predicted_churn.eq(predictions.score.ge(predictions.cutoff)).all(),
        "Frozen decisions differ",
    )
    path = ROOT / "results/current-model-diagnosis/october_cases.csv"
    verified_artifact(path.parent / "manifest.json", path)
    labels = pd.read_csv(path, usecols=study.KEYS + ["churn"])
    # The supplied-outcome join requires one unique strategy per target.
    # Derive this display key without changing frozen scores or decisions.
    predictions["strategy"] = predictions.minimum_precision_percent.map(
        lambda target: f"{NAME}_{target}"
    )
    cases = join_supplied_outcomes(predictions, labels)
    cases.to_csv(out / "evaluated_cases.csv", index=False)
    metrics = []
    for target, part in cases.groupby("minimum_precision_percent"):
        for group, rows in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            m = study.decision_metrics(rows.churn, rows.score, rows.cutoff)
            y, alert = rows.churn.eq(1), rows.predicted_churn.eq(1)
            counts = [
                int((y & alert).sum()),
                int((~y & alert).sum()),
                int((y & ~alert).sum()),
                int((~y & ~alert).sum()),
            ]
            require(
                counts == [m[k] for k in ["tp", "fp", "fn", "tn"]],
                "Independent confusion differs",
            )
            if not alert.any():
                m["precision"] = np.nan
            metrics.append(
                dict(
                    candidate=NAME,
                    minimum_precision_percent=int(target),
                    subgroup=group,
                    cases=len(rows),
                    precision_defined=bool(alert.any()),
                    recall_defined=bool(y.any()),
                    **m,
                )
            )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    previous = pd.read_csv(PRIOR / "evaluation/metrics.csv")
    combined = pd.concat(
        [previous[previous.minimum_precision_percent.isin(TARGETS)], metrics],
        ignore_index=True,
    )
    combined.to_csv(out / "combined_metrics.csv", index=False)
    names = dict(
        baseline="Original baseline",
        retained_logistic="Retained calibration",
        smooth_cadence="Smooth additive cadence",
        reseller_weighted="Reseller weighting",
        activity_experts="Activity experts",
        complementary_mean="Complementary blend",
        period_class_weighted="Period/class weighting",
        rbf_cadence="RBF cadence",
        early_quarter_share="Early-quarter share",
        guarded_component="Guarded component¹",
        positive_floor_veto="Positive-floor veto",
        cadence_regularized="Earlier cadence/regularization",
        cadence_unit_revenue="Earlier cadence/unit revenue",
        period_personalized_recency="Period/class + personalized recency",
    )
    report = "# Authorized personalized-recency October evaluation\n\nCells are test precision / recall (%). Targets are minimum DEVELOPMENT OOF precision; they do not guarantee test precision. Fixed three-fold seed78 grouped cutoff selection, unchanged model, predictions checkpointed before supplied outcomes. October previously inspected; descriptive reuse. Existing development acceptance remains failed.\n\n"
    report += (
        "| Procedure | "
        + " | ".join(f"{t}% target" for t in TARGETS)
        + " |\n|---|"
        + "---|" * len(TARGETS)
        + "\n"
    )
    for name, label in names.items():
        cells = []
        for target in TARGETS:
            row = combined[
                (combined.candidate == name)
                & (combined.minimum_precision_percent == target)
                & (combined.subgroup == "overall")
            ]
            if row.empty:
                cells.append("Infeasible")
            else:
                r = row.iloc[0]
                cells.append(
                    f"{100 * r.precision:.2f} / {100 * r.recall:.2f}"
                    if r.precision_defined
                    else f"Undefined / {100 * r.recall:.2f}"
                )
        report += "| " + label + " | " + " | ".join(cells) + " |\n"
    report += "\n¹ Guarded component selected the baseline fallback. Full development selections, confusion counts and activity subgroup metrics accompany this report. No deployment or test-driven cutoff revision.\n"
    (out / "report.md").write_text(report)
    write_json(
        out / "verification.json",
        dict(
            cutoffs_independently_enumerated=6,
            model_reload_scores_verified=20,
            independently_recounted_metric_rows=len(metrics),
            supplied_cases=492,
            supplied_churn_cases=int(labels.churn.sum()),
            test_cutoffs_selected=0,
        ),
    )
    write_manifest(
        out,
        dict(
            stage="evaluated",
            inputs=input_identity(
                dict(
                    predictions=output / "manifest.json",
                    outcomes=path,
                    old_evaluation=PRIOR / "evaluation/manifest.json",
                )
            ),
        ),
    )
    verify_manifest(out)
    print(metrics[metrics.subgroup.eq("overall")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else evaluate)(args.output)

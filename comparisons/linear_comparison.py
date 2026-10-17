"""Fixed linear learner against cached, matched HGB development predictions."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.matched_evaluation import evaluate_predictions
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)
from reseller_churn.modeling.model_factory import ModelFactory

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]
CANDIDATE = "linear_ten"


def fit_linear(train, validation_features):
    require(train.churn.nunique() == 2, "Training requires both classes")
    model = ModelFactory.create_logistic_regression(c=1.0, max_iter=2000)
    with threadpool_limits(limits=1):
        model.fit(
            train[study.FEATURES],
            train.churn,
            model__sample_weight=compute_sample_weight("balanced", train.churn),
        )
        scores = model.predict_proba(validation_features[study.FEATURES])[:, 1]
    imputer, scaler, classifier = (
        model.named_steps[name] for name in ("imputer", "scaler", "model")
    )
    require(classifier.n_iter_.max() < 2000, "Linear convergence was not established")
    content = pd.util.hash_pandas_object(
        train[[*KEYS, "churn", *study.FEATURES]], index=False
    ).values.tobytes()
    identity = {
        "candidate": CANDIDATE,
        "training_sha256": hashlib.sha256(content).hexdigest(),
        "estimator_hash": joblib.hash(model),
        "imputer_statistics": [
            float(x) if np.isfinite(x) else None for x in imputer.statistics_
        ],
        "missing_indicator_columns": imputer.indicator_.features_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_var": scaler.var_.tolist(),
        "coefficient_names": imputer.get_feature_names_out(study.FEATURES).tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "parameters": classifier.get_params(),
        "iterations": int(classifier.n_iter_.max()),
        "seed_score_range": 0.0,
        "training_rows": len(train),
        "training_positives": int(train.churn.sum()),
    }
    return scores, identity


def run(source, output):
    require(not output.exists(), "Preserve existing output")
    verify_manifest(source)
    previous = json.loads((source / "protocol.json").read_text())
    require(previous["repeat_seeds"] == [301, 919, 1201], "Unexpected repeats")
    data = read_csv(source / "development_input.csv")
    history_dir = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_dir)
    history = read_csv(history_dir / "case_histories.csv")
    profiles = history[[*KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol = ROOT / "comparisons/linear_comparison_protocol.md"
    shutil.copyfile(protocol, out / "protocol.md")
    shutil.copyfile(source / "development_input.csv", out / "development_input.csv")
    candidate = {
        "id": CANDIDATE,
        "features": study.FEATURES,
        "parameters": {
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": None,
            "solver": "lbfgs",
        },
        "preprocessing": "training median, missing indicators, standard scaling",
    }
    write_json(
        out / "protocol.json",
        {
            "description": "Fixed ten-feature linear procedure",
            "arms": [previous["arms"][0], candidate],
            "splits": previous["splits"],
            "repeat_seeds": previous["repeat_seeds"],
            "interval_tail_probability": 0.05 / 40,
            "new_fits": 60,
            "cached_baseline_fits": 300,
            "october_outcomes_accessed": False,
        },
    )
    parts = []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        source_fold = source / relative
        verify_manifest(source_fold)
        folder = new_output_directory(out / relative)
        inner = read_csv(source_fold / "inner_scores.csv")
        baseline_inner = inner[inner.candidate.eq("baseline")].copy()
        assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
            pd.MultiIndex.from_frame(train[KEYS])
        )
        require(assignment.notna().all(), "Incomplete inner assignment")
        inner_parts = []
        identities = []
        for fold in sorted(assignment.unique()):
            ti, vi = (
                np.flatnonzero(assignment.to_numpy() != fold),
                np.flatnonzero(assignment.to_numpy() == fold),
            )
            require(
                set(train.iloc[ti].StoreID).isdisjoint(train.iloc[vi].StoreID),
                "Inner reseller overlap",
            )
            scores, identity = fit_linear(
                train.iloc[ti], train.iloc[vi][study.FEATURES]
            )
            rows = train.iloc[vi][KEYS + ["churn"]].copy()
            rows["score"], rows["candidate"], rows["inner_fold"] = (
                scores,
                CANDIDATE,
                int(fold),
            )
            inner_parts.append(rows)
            identities.append({**identity, "stage": "inner", "inner_fold": int(fold)})
        candidate_inner = pd.concat(inner_parts, ignore_index=True)
        inner = pd.concat([baseline_inner, candidate_inner], ignore_index=True)
        selections = []
        for arm in ("baseline", CANDIDATE):
            rows = inner[inner.candidate.eq(arm)]
            selections.append(
                {
                    "candidate": arm,
                    **study.select_threshold(
                        rows.churn.to_numpy(), rows.score.to_numpy()
                    ),
                }
            )
        baseline = read_csv(source_fold / "predictions.csv")
        baseline = baseline[baseline.strategy.eq("baseline")]
        require(
            baseline.cutoff.eq(selections[0]["cutoff"]).all(),
            "Cached baseline cutoff differs",
        )
        scores, identity = fit_linear(train, valid[study.FEATURES])
        identities.append({**identity, "stage": "outer"})
        predicted = valid[KEYS + ["churn", "n_orders_3m"]].copy()
        predicted["score"], predicted["cutoff"], predicted["strategy"] = (
            scores,
            selections[1]["cutoff"],
            CANDIDATE,
        )
        predicted["repeat"], predicted["outer_fold"] = (
            split["repeat"],
            split["outer_fold"],
        )
        outer = pd.concat([baseline, predicted], ignore_index=True)
        parts.append(outer)
        outer.to_csv(folder / "predictions.csv", index=False)
        inner.to_csv(folder / "inner_scores.csv", index=False)
        pd.DataFrame(selections).to_csv(folder / "inner_selections.csv", index=False)
        write_json(folder / "model_identities.json", identities)
        shutil.copyfile(source_fold / "split.json", folder / "split.json")
        write_manifest(folder, {"new_linear_fits": 4, "cached_hgb_fits": 20})
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        data, predictions, profiles, out, CANDIDATE, 0.05 / 40
    )
    write_json(
        out / "baseline_fidelity.json",
        {
            "status": "unchanged cached HGB score/cutoff objective",
            "changed_baseline_decisions": 0,
        },
    )
    write_manifest(
        out,
        {
            "stage": "linear_comparison",
            "new_linear_fits": 60,
            "cached_hgb_fits": 300,
            "decision": decision,
            "inputs": input_identity(
                {
                    "cached_study_manifest": source / "manifest.json",
                    "protocol": protocol,
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "factory": ROOT / "reseller_churn/modeling/model_factory.py",
                    "analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )
    print(json.dumps(decision, indent=2))


def verify(directory):
    manifest = verify_manifest(directory)
    source = Path(manifest["inputs"]["cached_study_manifest"]["path"]).parent
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    checked = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        for arm in ("baseline", CANDIDATE):
            inside = inner[inner.candidate.eq(arm)].sort_values(["snapshot", "StoreID"])
            outside = outer[outer.strategy.eq(arm)].sort_values(["snapshot", "StoreID"])
            pd.testing.assert_frame_equal(
                inside[KEYS + ["churn"]].reset_index(drop=True),
                train.sort_values(["snapshot", "StoreID"])[
                    KEYS + ["churn"]
                ].reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                outside[KEYS + ["churn"]].reset_index(drop=True),
                valid.sort_values(["snapshot", "StoreID"])[
                    KEYS + ["churn"]
                ].reset_index(drop=True),
            )
            require(
                inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner reseller overlap",
            )
            require(
                outside.cutoff.eq(
                    independent_cutoff(inside.churn.to_numpy(), inside.score.to_numpy())
                ).all(),
                "Inner cutoff differs",
            )
        for identity in json.loads((folder / "model_identities.json").read_text()):
            fitted = train
            if identity["stage"] == "inner":
                held = inner.loc[
                    inner.candidate.eq(CANDIDATE)
                    & inner.inner_fold.eq(identity["inner_fold"]),
                    "StoreID",
                ]
                fitted = train[~train.StoreID.isin(held)]
            content = pd.util.hash_pandas_object(
                fitted[[*KEYS, "churn", *study.FEATURES]], index=False
            ).values.tobytes()
            require(
                hashlib.sha256(content).hexdigest() == identity["training_sha256"],
                "Training hash differs",
            )
            medians = fitted[study.FEATURES].median().to_numpy()
            np.testing.assert_allclose(
                identity["imputer_statistics"], medians, rtol=0, atol=0
            )
            missing = fitted[study.FEATURES].isna().to_numpy()
            x = fitted[study.FEATURES].to_numpy().copy()
            x = np.where(missing, medians, x)
            indicators = np.flatnonzero(missing.any(axis=0))
            require(
                indicators.tolist() == identity["missing_indicator_columns"],
                "Indicator set differs",
            )
            x = np.column_stack([x, missing[:, indicators]])
            np.testing.assert_allclose(
                identity["scaler_mean"], x.mean(axis=0), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                identity["scaler_var"], x.var(axis=0), rtol=1e-12, atol=1e-12
            )
            require(
                identity["parameters"]["C"] == 1.0
                and identity["parameters"]["class_weight"] is None
                and identity["iterations"] < 2000,
                "Linear procedure differs or failed convergence",
            )
            checked += 1
    predictions = read_csv(directory / "validation_predictions.csv")
    metrics = pd.read_csv(directory / "metrics.csv")
    count = 0
    for scope, columns in [
        ("repeat_pooled", ["repeat", "strategy"]),
        ("outer_fold", ["repeat", "strategy", "outer_fold"]),
    ]:
        for identity, part in predictions.groupby(columns):
            selector = metrics.scope.eq(scope) & metrics.slice.eq("overall")
            for key, value in zip(columns, identity):
                selector &= metrics["policy" if key == "strategy" else key].eq(value)
            row = metrics[selector].squeeze()
            a, y = part.score.ge(part.cutoff), part.churn.eq(1)
            require(
                [int(mask.sum()) for mask in [a & y, a & ~y, ~a & y, ~a & ~y]]
                == [row.tp, row.fp, row.fn, row.tn],
                "Metric counts differ",
            )
            count += 1
    result = {
        "status": "passed",
        "outer_splits": 15,
        "inner_cutoffs_independently_reproduced": 30,
        "training_imputer_scaler_identities": checked,
        "overall_metric_rows": count,
        "models_fitted_by_verifier": 0,
        "october_outcomes_accessed": False,
    }
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/cadence-stability-20261016"
    )
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        run(args.source, args.output)

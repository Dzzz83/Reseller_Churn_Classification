"""One fixed additive spline learner on the established cadence inputs."""

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.breadth_reporting import markdown_table
from comparisons.crossfit_percentile import FEATURES
from comparisons.development_reporting import cutoff_transfer_tables
from comparisons.fixed_feature_comparison import evaluate_fixed_fold
from comparisons.matched_evaluation import evaluate_predictions
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
CANDIDATE = "smooth_cadence"
TAIL = 0.000125


def fit_smooth(train, validation_features, spec, *, retain_models=False):
    """Fit every transform on training rows; held labels are not accepted."""
    numeric = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        SplineTransformer(
            n_knots=5,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        ),
    )
    features = FeatureUnion(
        [("numeric", numeric), ("missing", MissingIndicator(features="all"))]
    )
    model = make_pipeline(
        features,
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-8),
    )
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(
            train[FEATURES],
            train.churn,
            logisticregression__sample_weight=compute_sample_weight(
                "balanced", train.churn
            ),
        )
        scores = model.predict_proba(validation_features[FEATURES])[:, 1]
    imputer, splines = numeric.steps[0][1], numeric.steps[1][1]
    scaler, logistic = model.steps[1][1], model.steps[2][1]
    content = pd.util.hash_pandas_object(
        train[KEYS + ["churn"] + FEATURES], index=False
    ).values.tobytes()
    identity = dict(
        candidate=spec["id"],
        training_sha256=hashlib.sha256(content).hexdigest(),
        training_rows=len(train),
        training_positives=int(train.churn.sum()),
        imputer_statistics=imputer.statistics_.tolist(),
        splines=[
            dict(knots=b.t.tolist(), coefficients=b.c.tolist(), degree=b.k)
            for b in splines.bsplines_
        ],
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        coefficients=logistic.coef_[0].tolist(),
        intercept=float(logistic.intercept_[0]),
        iterations=int(logistic.n_iter_[0]),
    )
    return scores, [identity], 0.0, [model] if retain_models else []


def run(output):
    require(not output.exists(), "Preserve existing output")
    source = ROOT / "results/cadence-stability-20261016"
    verify_manifest(source)
    previous = json.loads((source / "protocol.json").read_text())
    data = read_csv(source / "development_input.csv")
    history_dir = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_dir)
    history = read_csv(history_dir / "case_histories.csv")
    profiles = history[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    spec = dict(
        id=CANDIDATE,
        features=FEATURES,
        description="Fixed smooth additive cadence learner",
        parameters=dict(
            C=1.0, knots=5, degree=3, extrapolation="constant", max_iter=2000, tol=1e-8
        ),
    )
    out = new_output_directory(output)
    protocol_path = ROOT / "comparisons/smooth_cadence_protocol.md"
    shutil.copyfile(protocol_path, out / "protocol.md")
    shutil.copyfile(Path(__file__), out / "as_run_smooth_cadence.py")
    data.to_csv(out / "development_input.csv", index=False)
    protocol = dict(
        arms=[previous["arms"][0], spec],
        splits=previous["splits"],
        repeat_seeds=previous["repeat_seeds"],
        interval_tail_probability=TAIL,
        period_limits=LIMITS,
        alpha_spending_cycle=2,
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "prefit_identity.json",
        dict(
            inputs=input_identity(
                {
                    "source": source / "manifest.json",
                    "history": history_dir / "manifest.json",
                    "protocol": protocol_path,
                }
            ),
            implementation=input_identity(
                {
                    "runner": Path(__file__),
                    "fitting": ROOT / "comparisons/fixed_feature_comparison.py",
                    "assessment": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        ),
    )
    parts, transfers = [], []
    for split in previous["splits"]:
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        outer = evaluate_fixed_fold(
            data,
            split,
            source / relative,
            out / relative,
            spec,
            fit_function=fit_smooth,
        )
        inner = read_csv(out / relative / "inner_scores.csv")
        transfers.append(cutoff_transfer_tables(inner, outer, data))
        parts.append(outer)
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    pd.concat(transfers, ignore_index=True).to_csv(
        out / "cutoff_transfer.csv", index=False
    )
    evaluate_predictions(
        data, predictions, profiles, out, CANDIDATE, TAIL, period_limits=LIMITS
    )
    decision = intersection_gate(out)
    metrics = pd.read_csv(out / "operating_metrics.csv")
    overall = metrics[metrics.scope.eq("repeat_pooled") & metrics.slice.eq("overall")]
    (out / "report.md").write_text(
        "# Smooth additive cadence DEVELOPMENT comparison\n\nAcceptance: "
        + str(decision["accepted"])
        + ". 60 deterministic fits; no HGB baseline refits.\n\n"
        + markdown_table(
            overall[
                [
                    "strategy",
                    "repeat",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "recall",
                    "precision",
                    "alert_count",
                ]
            ]
        )
        + "\n\nAll folds,periods,activity/intersections: operating_metrics.csv. Complete guards: decision.json. "
        "Ranking and outcome-derived tied recall points are descriptive. Conditional reseller intervals omit "
        "retraining,reselection,adaptive development and future calendar/population changes.\n"
    )
    write_manifest(
        out,
        dict(
            stage="smooth_additive_cadence",
            decision=decision,
            new_fits=60,
            inputs=input_identity({"source": source / "manifest.json"}),
        ),
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

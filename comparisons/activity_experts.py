"""Two fixed activity experts retaining the original full-partition row weights."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.breadth_reporting import markdown_table
from comparisons.fixed_feature_comparison import run_fixed_hgb
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import input_identity, verify_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "activity_experts"
TAIL = 0.00003125


def training_hash(rows, features):
    values = pd.util.hash_pandas_object(
        rows[study.KEYS + ["churn"] + features], index=False
    ).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def fit_experts(train, validation_features, spec, *, retain_models=False):
    """Both training routes are retained; validation labels are not accepted."""
    weights = compute_sample_weight("balanced", train.churn)
    recent = train.n_orders_3m.gt(0).to_numpy()
    held_recent = validation_features.n_orders_3m.gt(0).to_numpy()
    for route in [False, True]:
        require(
            train.loc[recent == route, "churn"].nunique() == 2,
            "Infeasible expert: each fitting activity route requires both classes",
        )
    records, predictions, models = [], [], []
    full_digest = training_hash(train, spec["features"])
    with threadpool_limits(limits=1):
        for seed in study.MODEL_SEEDS:
            scores = np.full(len(validation_features), np.nan)
            for route in [False, True]:
                subset = train.loc[recent == route]
                held = validation_features.loc[held_recent == route]
                route_weights = weights[recent == route]
                model_features = [
                    name for name in spec["features"] if subset[name].notna().any()
                ]
                require(bool(model_features), "No observed expert features")
                model = HistGradientBoostingClassifier(
                    **spec["parameters"], random_state=seed
                )
                model.fit(
                    subset[model_features], subset.churn, sample_weight=route_weights
                )
                if retain_models:
                    models.append(
                        dict(
                            seed=seed,
                            recent_purchase=route,
                            features=model_features,
                            model=model,
                        )
                    )
                component = (
                    model.predict_proba(held[model_features])[:, 1]
                    if len(held)
                    else np.array([])
                )
                scores[held_recent == route] = component
                keys = held[study.KEYS].copy()
                keys["snapshot"] = keys.snapshot.dt.strftime("%Y-%m-%d")
                records.append(
                    dict(
                        candidate=spec["id"],
                        seed=int(seed),
                        recent_purchase=route,
                        training_sha256=full_digest,
                        expert_training_sha256=training_hash(subset, spec["features"]),
                        model_features=model_features,
                        omitted_all_missing=[
                            name
                            for name in spec["features"]
                            if name not in model_features
                        ],
                        training_rows=len(subset),
                        training_positives=int(subset.churn.sum()),
                        weights=route_weights.tolist(),
                        parameters=model.get_params(),
                        estimator_hash=joblib.hash(model),
                        validation_cases=keys.to_dict("records"),
                        validation_scores=component.tolist(),
                    )
                )
            require(np.isfinite(scores).all(), "Routing left unscored rows")
            predictions.append(scores)
    return (
        np.mean(predictions, axis=0),
        records,
        float(np.ptp(predictions, axis=0).max()) if len(validation_features) else 0.0,
        models,
    )


def run(output):
    prior = ROOT / "results/reseller-weighting-20261018/decision.json"
    require(
        prior.is_file() and not json.loads(prior.read_text())["accepted"],
        "Contingent trial requires failed cycle3",
    )
    source = ROOT / "results/cadence-stability-20261016"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(source)
    verify_manifest(history_source)
    data = read_csv(source / "development_input.csv")
    history = read_csv(history_source / "case_histories.csv")
    profiles = history[study.KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    spec = dict(
        id=CANDIDATE,
        features=study.FEATURES,
        parameters=study.BASE_PARAMETERS,
        description="Fixed activity routing with original balanced full-partition weights",
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/activity_experts_protocol.md",
        TAIL,
        dict(history=history_source / "manifest.json", entrypoint=Path(__file__)),
        period_limits=LIMITS,
        fit_function=fit_experts,
        post_assessment=intersection_gate,
    )
    metrics = pd.read_csv(output / "operating_metrics.csv")
    rows = metrics[metrics.scope.eq("repeat_pooled") & metrics.slice.eq("overall")]
    (output / "report.md").write_text(
        "# Activity expert DEVELOPMENT comparison\n\n"
        + markdown_table(
            rows[
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
        + "\n\nAcceptance: "
        + str(json.loads((output / "decision.json").read_text())["accepted"])
        + ". All folds/periods/groups/intersections and undefined flags are preserved. "
        "600 expert fits, original features/parameters/full-partition weights; no baseline refit. "
        "Conditional intervals omit retraining,adaptive selection and future shifts.\n"
    )
    write_json(
        output / "report_identity.json",
        input_identity({"report": output / "report.md"}),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

"""Original ten-feature HGB with fitting-only inverse reseller-history weights."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.breadth_reporting import markdown_table
from comparisons.fixed_feature_comparison import run_fixed_hgb
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import input_identity, verify_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "reseller_weighted"
TAIL = 0.0000625


def reseller_weights(train):
    """Balance class totals after equal reseller influence before class adjustment."""
    require(
        train.churn.nunique() == 2 and train.churn.isin([0, 1]).all(),
        "Two binary classes required",
    )
    counts = train.groupby("StoreID").StoreID.transform("size").to_numpy()
    inverse = 1.0 / counts
    labels = train.churn.to_numpy()
    totals = np.bincount(labels, weights=inverse, minlength=2)
    return inverse * len(train) / (2 * totals[labels])


def fit_weighted(
    train,
    validation_features,
    spec,
    *,
    weight_function=reseller_weights,
    retain_models=False,
):
    weights = weight_function(train)
    content = pd.util.hash_pandas_object(
        train[study.KEYS + ["churn"] + spec["features"]], index=False
    ).values.tobytes()
    records, scores, models = [], [], []
    with threadpool_limits(limits=1):
        for seed in study.MODEL_SEEDS:
            model = HistGradientBoostingClassifier(
                **spec["parameters"], random_state=seed
            )
            model.fit(train[spec["features"]], train.churn, sample_weight=weights)
            scores.append(
                model.predict_proba(validation_features[spec["features"]])[:, 1]
            )
            if retain_models:
                models.append(model)
            records.append(
                dict(
                    candidate=spec["id"],
                    seed=int(seed),
                    training_sha256=hashlib.sha256(content).hexdigest(),
                    estimator_hash=joblib.hash(model),
                    weights=weights.tolist(),
                    weight_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                    parameters=model.get_params(),
                    training_rows=len(train),
                    training_positives=int(train.churn.sum()),
                )
            )
    return np.mean(scores, axis=0), records, float(np.ptp(scores, axis=0).max()), models


def run(output):
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
        description="Original ten-feature HGB with inverse-history class-balanced weights",
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/reseller_weighting_protocol.md",
        TAIL,
        dict(history=history_source / "manifest.json", entrypoint=Path(__file__)),
        period_limits=LIMITS,
        fit_function=fit_weighted,
        post_assessment=intersection_gate,
    )
    # Reporting is appended after the immutable fitting/assessment manifest.
    metrics = pd.read_csv(output / "operating_metrics.csv")
    rows = metrics[metrics.scope.eq("repeat_pooled") & metrics.slice.eq("overall")]
    (output / "report.md").write_text(
        "# Reseller-weighted DEVELOPMENT comparison\n\n"
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
        + ". All guards,intervals and limits: decision.json. All folds/groups/intersections "
        "and undefined flags: operating_metrics.csv. Ranking/tied outcome-derived points are "
        "descriptive; conditional uncertainty omits refitting,selection and future shifts.\n"
    )
    write_json(
        output / "report_identity.json",
        input_identity({"report": output / "report.md"}),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

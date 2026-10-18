"""One fixed smooth multivariate SVM margin boundary on established cadence inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.crossfit_percentile import FEATURES
from comparisons.fixed_feature_comparison import run_fixed_hgb
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "rbf_cadence"
TAIL = 0.00000390625
PARAMETERS = dict(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight=None,
    probability=False,
    tol=0.001,
    max_iter=-1,
    shrinking=True,
    cache_size=200,
)


def fit_margin(train, validation_features, spec, *, retain_models=False):
    """Transform fitting rows only; margin scores require no internal probability CV."""
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        SVC(**PARAMETERS),
    )
    weights = compute_sample_weight("balanced", train.churn)
    with threadpool_limits(limits=1):
        model.fit(train[FEATURES], train.churn, svc__sample_weight=weights)
        margin = model.decision_function(validation_features[FEATURES])
    imputer, scaler, svm = [step[1] for step in model.steps]
    require(svm.fit_status_ == 0, "SVM fit did not converge")
    content = pd.util.hash_pandas_object(
        train[study.KEYS + ["churn"] + FEATURES], index=False
    ).values.tobytes()
    identity = dict(
        candidate=spec["id"],
        training_sha256=hashlib.sha256(content).hexdigest(),
        training_rows=len(train),
        training_positives=int(train.churn.sum()),
        parameters=svm.get_params(),
        fit_status=int(svm.fit_status_),
        iterations=svm.n_iter_.tolist(),
        imputer_statistics=imputer.statistics_.tolist(),
        indicator_columns=imputer.indicator_.features_.tolist(),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        support_indices=svm.support_.tolist(),
        support_vectors=svm.support_vectors_.tolist(),
        dual_coefficients=svm.dual_coef_[0].tolist(),
        intercept=float(svm.intercept_[0]),
        effective_gamma=float(svm._gamma),
        weights=weights.tolist(),
    )
    return expit(margin), [identity], 0.0, [model] if retain_models else []


def run(output):
    prior = ROOT / "results/period-weighting-20261018/decision.json"
    require(
        prior.is_file() and not json.loads(prior.read_text())["accepted"],
        "Contingent procedure requires failed cycle6",
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
        features=FEATURES,
        parameters=PARAMETERS,
        description="Fixed RBF margin learner on established cadence inputs; scores are not probabilities",
    )
    run_fixed_hgb(
        data,
        profiles,
        source,
        output,
        spec,
        ROOT / "comparisons/rbf_cadence_protocol.md",
        TAIL,
        dict(history=history_source / "manifest.json", entrypoint=Path(__file__)),
        period_limits=LIMITS,
        fit_function=fit_margin,
        post_assessment=intersection_gate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

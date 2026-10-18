"""Smooth learner preprocessing remains training-only and reconstructible."""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import expit

from comparisons.crossfit_percentile import FEATURES
from comparisons.smooth_cadence import fit_smooth


def reconstruct(frame, identity):
    values = frame[FEATURES].to_numpy(dtype=float)
    missing = np.isnan(values)
    filled = np.where(missing, identity["imputer_statistics"], values)
    columns = []
    for index, spec in enumerate(identity["splines"]):
        spline = BSpline.construct_fast(
            np.asarray(spec["knots"], dtype=float),
            np.asarray(spec["coefficients"], dtype=float),
            spec["degree"],
        )
        low, high = spline.t[spline.k], spline.t[-spline.k - 1]
        columns.append(spline(np.clip(filled[:, index], low, high))[:, :-1])
    design = np.concatenate([*columns, missing.astype(float)], axis=1)
    standardized = (design - identity["scaler_mean"]) / identity["scaler_scale"]
    return expit(
        standardized @ np.array(identity["coefficients"]) + identity["intercept"]
    )


def fixture():
    rng = np.random.default_rng(23)
    train = pd.DataFrame(rng.normal(size=(160, len(FEATURES))), columns=FEATURES)
    train["StoreID"] = np.repeat(range(1, 81), 2)
    train["snapshot"] = pd.to_datetime(["2013-01-01", "2013-04-01"] * 80)
    train["churn"] = (train.recency_days.gt(0.3)).astype(int)
    train.loc[::3, "gap_cv"] = np.nan
    train["n_orders_3m"] = 1.0  # constant knots and zero-variance basis columns
    return train


def test_synthetic_spline_smoke_missing_constant_and_extrapolation():
    train = fixture()
    held = train.iloc[:8][FEATURES].copy()
    held.iloc[0, 1] = 1e10
    held.iloc[1, 1] = -1e10
    scores, identities, spread, _ = fit_smooth(train, held, {"id": "synthetic"})
    assert spread == 0 and len(identities) == 1
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, reconstruct(held, identities[0]), atol=1e-14)
    assert (
        identities[0]["imputer_statistics"][FEATURES.index("gap_cv")]
        == train.gap_cv.median()
    )
    assert len(identities[0]["splines"]) == 12


def test_spline_nested_fold_callback_preserves_cached_baseline(tmp_path):
    import json

    from comparisons.basket_breadth import evaluate_fold
    from comparisons.fixed_feature_comparison import evaluate_fixed_fold
    from comparisons.grouped_precision_recall import (
        grouped_splits,
        synthetic_development,
    )

    data = synthetic_development()
    data["last_order_lines"] = 1.0
    train, held = grouped_splits(data, 5, 42)[0]
    split = dict(
        repeat=42,
        outer_fold=0,
        train_rows=train.tolist(),
        validation_rows=held.tolist(),
        train_store_ids=sorted(data.iloc[train].StoreID.unique().tolist()),
        validation_store_ids=sorted(data.iloc[held].StoreID.unique().tolist()),
    )
    expected = evaluate_fold(data, split, tmp_path / "source")
    out = tmp_path / "candidate"
    actual = evaluate_fixed_fold(
        data,
        split,
        tmp_path / "source/repeat_42/fold_0",
        out,
        {"id": "smooth", "features": FEATURES},
        fit_function=fit_smooth,
    )
    pd.testing.assert_frame_equal(
        actual[actual.strategy.eq("baseline")].reset_index(drop=True),
        expected[expected.strategy.eq("baseline")].reset_index(drop=True),
    )
    records = json.loads((out / "model_identities.json").read_text())
    assert len(records) == 24
    assert sum(r["candidate"] == "smooth" for r in records) == 4
    assert np.isfinite(actual.score).all()

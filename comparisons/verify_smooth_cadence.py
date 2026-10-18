"""Reconstruct saved spline predictions and fitting statistics without refits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import expit

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.crossfit_percentile import FEATURES
from comparisons.verify_basket_breadth import independent_cutoff
from comparisons.verify_retained_calibration import confusion, ordered
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
CANDIDATE = "smooth_cadence"


def saved_design(frame, identity):
    values = frame[FEATURES].to_numpy(dtype=float)
    missing = np.isnan(values)
    filled = np.where(missing, identity["imputer_statistics"], values)
    columns = []
    for index, spec in enumerate(identity["splines"]):
        # construct_fast preserves constant/duplicate knots used by sklearn.
        spline = BSpline.construct_fast(
            np.asarray(spec["knots"], dtype=float),
            np.asarray(spec["coefficients"], dtype=float),
            spec["degree"],
        )
        low, high = spline.t[spline.k], spline.t[-spline.k - 1]
        columns.append(spline(np.clip(filled[:, index], low, high))[:, :-1])
    return np.concatenate([*columns, missing.astype(float)], axis=1)


def reconstruct(frame, identity):
    design = saved_design(frame, identity)
    z = (design - identity["scaler_mean"]) / identity["scaler_scale"]
    return expit(z @ np.array(identity["coefficients"]) + identity["intercept"])


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        original = Path(identity["path"])
        archived = directory / "source_snapshot" / original.name
        path = archived if archived.is_file() else original
        require(file_sha256(path) == identity["sha256"], "Input/code identity changed")
    source = directory.parent / "cadence-stability-20261016"
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    predictions = read_csv(directory / "validation_predictions.csv")
    cached = read_csv(source / "validation_predictions.csv")
    pd.testing.assert_frame_equal(
        ordered(predictions[predictions.strategy.eq("baseline")]),
        ordered(cached[cached.strategy.eq("baseline")]),
    )
    checked = 0
    max_gradient = 0.0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        original = read_csv(source / relative / "inner_scores.csv")
        pd.testing.assert_frame_equal(
            ordered(inner[inner.candidate.eq("baseline")]),
            ordered(original[original.candidate.eq("baseline")]),
        )
        identities = json.loads((folder / "model_identities.json").read_text())
        for identity in [r for r in identities if r["candidate"] == CANDIDATE]:
            if identity["stage"] == "outer":
                fitting, held = train, valid
                saved = outer[outer.strategy.eq(CANDIDATE)]
            else:
                member = inner[
                    inner.candidate.eq("baseline")
                    & inner.inner_fold.eq(identity["inner_fold"])
                ]
                fitting, held = (
                    train[~train.StoreID.isin(member.StoreID)],
                    train[train.StoreID.isin(member.StoreID)],
                )
                saved = inner[
                    inner.candidate.eq(CANDIDATE)
                    & inner.inner_fold.eq(identity["inner_fold"])
                ]
            require(set(fitting.StoreID).isdisjoint(held.StoreID), "Reseller overlap")
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(
                    fitting[KEYS + ["churn"] + FEATURES], index=False
                ).values.tobytes()
            ).hexdigest()
            require(identity["training_sha256"] == digest, "Training hash differs")
            medians = fitting[FEATURES].median().fillna(0).to_numpy()
            np.testing.assert_array_equal(medians, identity["imputer_statistics"])
            values = fitting[FEATURES].fillna(dict(zip(FEATURES, medians))).to_numpy()
            expected = np.percentile(values, [0, 25, 50, 75, 100], axis=0)
            for j, spec in enumerate(identity["splines"]):
                np.testing.assert_array_equal(spec["knots"][3:-3], expected[:, j])
            design = saved_design(fitting, identity)
            np.testing.assert_allclose(
                design.mean(axis=0), identity["scaler_mean"], atol=1e-14
            )
            expected_scale = design.std(axis=0)
            expected_scale[expected_scale < 1e-12] = 1.0
            np.testing.assert_allclose(
                expected_scale, identity["scaler_scale"], atol=1e-12
            )
            aligned = held[KEYS].merge(saved, on=KEYS, validate="one_to_one")
            np.testing.assert_allclose(
                aligned.score, reconstruct(held, identity), rtol=1e-12, atol=1e-14
            )
            z = (design - identity["scaler_mean"]) / identity["scaler_scale"]
            y = fitting.churn.to_numpy()
            weights = len(y) / (2 * np.bincount(y)[y])
            residual = (
                expit(z @ identity["coefficients"] + identity["intercept"]) - y
            ) * weights
            gradient = max(
                abs(residual.mean()),
                np.max(
                    abs(
                        z.T @ residual / len(y)
                        + np.array(identity["coefficients"]) / len(y)
                    )
                ),
            )
            max_gradient = max(max_gradient, float(gradient))
            require(gradient < 2e-6, "Logistic optimum mismatch")
            checked += 1
        for arm in ["baseline", CANDIDATE]:
            inside = inner[inner.candidate.eq(arm)]
            require(
                outer[outer.strategy.eq(arm)]
                .cutoff.eq(independent_cutoff(inside.churn, inside.score))
                .all(),
                "Training-only cutoff differs",
            )
    metrics = pd.read_csv(directory / "operating_metrics.csv")
    for _, row in metrics.iterrows():
        rows = predictions[
            predictions.strategy.eq(row.strategy) & predictions.repeat.eq(row["repeat"])
        ]
        if row.scope == "outer_fold":
            rows = rows[rows.outer_fold.eq(row.outer_fold)]
        if row["slice"] in ["snapshot", "snapshot_activity"]:
            rows = rows[rows.snapshot.eq(row["group"].split("/")[0])]
        if row["slice"] in ["activity", "snapshot_activity"]:
            rows = rows[
                rows.n_orders_3m.gt(0).eq(
                    row["group"].split("/")[-1] == "recent_purchase"
                )
            ]
        require(
            confusion(rows) == [row[k] for k in ["tp", "fp", "fn", "tn"]],
            "Metric count differs",
        )
    result = dict(
        status="passed",
        training_identities=checked,
        cutoffs=30,
        operating_rows=len(metrics),
        maximum_logistic_gradient=max_gradient,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(
        not (directory / "independent_verification.json").exists(), "Preserve audit"
    )
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

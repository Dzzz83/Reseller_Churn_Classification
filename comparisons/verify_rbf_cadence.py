"""Reconstruct SVM kernels, fitting transforms, margins and cutoffs without refits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.crossfit_percentile import FEATURES
from comparisons.verify_basket_breadth import independent_cutoff
from comparisons.verify_retained_calibration import confusion
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
CANDIDATE = "rbf_cadence"


def design(frame, identity):
    raw = frame[FEATURES].to_numpy(dtype=float)
    missing = np.isnan(raw)
    filled = np.where(missing, identity["imputer_statistics"], raw)
    augmented = np.concatenate(
        [filled, missing[:, identity["indicator_columns"]]], axis=1
    )
    return (augmented - identity["scaler_mean"]) / identity["scaler_scale"]


def margins(frame, identity):
    transformed = design(frame, identity)
    kernel = np.exp(
        -identity["effective_gamma"]
        * cdist(transformed, identity["support_vectors"], "sqeuclidean")
    )
    return kernel @ np.array(identity["dual_coefficients"]) + identity["intercept"]


def verify(directory):
    verify_manifest(directory)
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in [*prefit["inputs"].values(), *prefit["implementation"].values()]:
        archived = directory / "source_snapshot" / Path(identity["path"]).name
        path = archived if archived.is_file() else Path(identity["path"])
        require(file_sha256(path) == identity["sha256"], "Pre-fit input/code changed")
    data = read_csv(directory / "development_input.csv")
    protocol = json.loads((directory / "protocol.json").read_text())
    checked = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        folder = directory / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        records = json.loads((folder / "model_identities.json").read_text())
        for record in [r for r in records if r["candidate"] == CANDIDATE]:
            if record["stage"] == "outer":
                fitting, held = train, valid
                saved = outer[outer.strategy.eq(CANDIDATE)]
            else:
                member = inner[
                    inner.candidate.eq("baseline")
                    & inner.inner_fold.eq(record["inner_fold"])
                ]
                fitting, held = (
                    train[~train.StoreID.isin(member.StoreID)],
                    train[train.StoreID.isin(member.StoreID)],
                )
                saved = inner[
                    inner.candidate.eq(CANDIDATE)
                    & inner.inner_fold.eq(record["inner_fold"])
                ]
            require(set(fitting.StoreID).isdisjoint(held.StoreID), "Reseller leakage")
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(
                    fitting[KEYS + ["churn"] + FEATURES], index=False
                ).values.tobytes()
            ).hexdigest()
            require(digest == record["training_sha256"], "Fitting hash mismatch")
            medians = fitting[FEATURES].median().fillna(0).to_numpy()
            np.testing.assert_array_equal(record["imputer_statistics"], medians)
            indicators = np.flatnonzero(fitting[FEATURES].isna().any(axis=0))
            np.testing.assert_array_equal(record["indicator_columns"], indicators)
            raw = fitting[FEATURES].to_numpy(dtype=float)
            augmented = np.concatenate(
                [np.where(np.isnan(raw), medians, raw), np.isnan(raw)[:, indicators]],
                axis=1,
            )
            np.testing.assert_allclose(
                augmented.mean(axis=0), record["scaler_mean"], atol=1e-13
            )
            scales = augmented.std(axis=0)
            scales[scales < 1e-12] = 1.0
            np.testing.assert_allclose(scales, record["scaler_scale"], atol=1e-12)
            standardized = design(fitting, record)
            np.testing.assert_array_equal(
                standardized[record["support_indices"]], record["support_vectors"]
            )
            y = fitting.churn.to_numpy()
            weights = len(fitting) / (2 * np.bincount(y)[y])
            np.testing.assert_array_equal(record["weights"], weights)
            dual = np.asarray(record["dual_coefficients"])
            support = np.asarray(record["support_indices"])
            require(
                np.all(np.abs(dual) <= weights[support] + 1e-9), "Weighted C dual bound"
            )
            require(np.all(np.sign(dual) == 2 * y[support] - 1), "Support label signs")
            require(abs(dual.sum()) < 1e-7, "Dual equality constraint")
            np.testing.assert_allclose(
                record["effective_gamma"],
                1 / (standardized.shape[1] * standardized.var()),
                atol=1e-15,
            )
            aligned = held[KEYS].merge(saved, on=KEYS, validate="one_to_one")
            np.testing.assert_allclose(
                aligned.score, expit(margins(held, record)), atol=2e-13, rtol=1e-11
            )
            require(
                record["fit_status"] == 0 and not record["parameters"]["probability"],
                "Unexpected SVM fitting/calibration",
            )
            checked += 1
        for arm in ["baseline", CANDIDATE]:
            inside = inner[inner.candidate.eq(arm)]
            require(
                outer[outer.strategy.eq(arm)]
                .cutoff.eq(independent_cutoff(inside.churn, inside.score))
                .all(),
                "Cutoff mismatch",
            )
    metrics = pd.read_csv(directory / "operating_metrics.csv")
    predictions = read_csv(directory / "validation_predictions.csv")
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
            "Operating count mismatch",
        )
    result = dict(
        status="passed",
        support_training_identities=checked,
        cutoffs=30,
        operating_rows=len(metrics),
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    require(
        not (directory / "independent_verification.json").exists(),
        "Preserve verification",
    )
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

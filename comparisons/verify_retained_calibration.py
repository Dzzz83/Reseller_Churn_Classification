"""Independent saved-score/count audit; never fits or loads an estimator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.crossfit_percentile import FEATURES
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]
CANDIDATE = "retained_logistic"


def ordered(frame):
    return frame.sort_values(KEYS).reset_index(drop=True)


def confusion(rows):
    positive, alert = rows.churn.eq(1), rows.score.ge(rows.cutoff)
    return [
        int(x.sum())
        for x in [
            positive & alert,
            ~positive & alert,
            positive & ~alert,
            ~positive & ~alert,
        ]
    ]


def verify(directory):
    verify_manifest(directory)
    child = directory / CANDIDATE
    completion = json.loads((child / "completion_manifest.json").read_text())
    for name, digest in completion["outputs"].items():
        require(file_sha256(child / name) == digest, "Completion artifact changed")
    source = directory.parent / "crossfit-percentile-20261017"
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    prefit = json.loads((directory / "prefit_identity.json").read_text())
    for identity in prefit["inputs"].values():
        require(
            file_sha256(Path(identity["path"])) == identity["sha256"],
            "Pre-fit identity changed",
        )
    require(
        file_sha256(directory / "as_run_retained_calibration.py")
        == prefit["implementation"]["runner"]["sha256"],
        "As-run runner changed",
    )
    require(
        file_sha256(Path(prefit["implementation"]["assessment"]["path"]))
        == prefit["implementation"]["assessment"]["sha256"],
        "Assessment changed",
    )
    predictions = read_csv(directory / "validation_predictions.csv")
    require(len(predictions) == 8250, "Incomplete matched population")
    original_predictions = read_csv(source / "validation_predictions.csv")
    pd.testing.assert_frame_equal(
        ordered(predictions[predictions.strategy.eq("baseline")]),
        ordered(original_predictions[original_predictions.strategy.eq("baseline")]),
    )
    counts = 0
    maximum_gradient = 0.0
    negative_slopes = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        baseline_inner = inner[inner.candidate.eq("baseline")]
        original_inner = read_csv(source / relative / "inner_scores.csv")
        pd.testing.assert_frame_equal(
            ordered(baseline_inner),
            ordered(original_inner[original_inner.candidate.eq("baseline")]),
        )
        records = json.loads((folder / "model_identities.json").read_text())
        original_records = json.loads(
            (source / relative / "model_identities.json").read_text()
        )
        require(len(records) == len(original_records) == 40, "Component count")
        for record in records:
            original = next(
                r
                for r in original_records
                if all(
                    r.get(k) == record.get(k)
                    for k in ["stage", "inner_fold", "half", "seed"]
                )
            )
            require(
                all(record[k] == v for k, v in original.items()),
                "Cached identity changed",
            )
            if record["stage"] == "outer":
                base, held = train, valid
            else:
                stores = baseline_inner.loc[
                    baseline_inner.inner_fold.eq(record["inner_fold"]), "StoreID"
                ]
                base, held = (
                    train[~train.StoreID.isin(stores)],
                    train[train.StoreID.isin(stores)],
                )
            refs = pd.DataFrame(record["reference_cases"])
            refs["snapshot"] = pd.to_datetime(refs.snapshot)
            reference = refs.merge(data, on=KEYS, validate="one_to_one", how="left")
            require(reference.churn.notna().all(), "Unknown reference case")
            reference_stores = set(reference.StoreID)
            fitting = base[~base.StoreID.isin(reference_stores)]
            require(
                reference_stores.isdisjoint(held.StoreID)
                and set(base.StoreID).isdisjoint(held.StoreID),
                "Reference/held leakage",
            )
            pd.testing.assert_frame_equal(
                ordered(reference[KEYS]),
                ordered(base[base.StoreID.isin(reference_stores)][KEYS]),
            )
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(
                    fitting[KEYS + ["churn"] + FEATURES], index=False
                ).values.tobytes()
            ).hexdigest()
            require(
                digest == record["training_sha256"], "Independent fitting hash mismatch"
            )
            require(
                reference.churn.tolist() == record["reference_labels"],
                "Reference labels differ",
            )
            probability = np.clip(
                np.array(record["validation_raw_scores"]), 1e-12, 1 - 1e-12
            )
            logits = np.log(probability) - np.log1p(-probability)
            mapped = expit(record["slope"] * logits + record["intercept"])
            np.testing.assert_allclose(
                mapped, record["mapped_scores"], rtol=1e-12, atol=1e-14
            )
            ref_probability = np.clip(
                np.array(record["reference_scores"]), 1e-12, 1 - 1e-12
            )
            x = np.log(ref_probability) - np.log1p(-ref_probability)
            residual = (
                expit(record["slope"] * x + record["intercept"])
                - reference.churn.to_numpy()
            )
            gradient = max(
                abs(residual.mean()),
                abs((residual * x).mean() + record["slope"] / len(x)),
            )
            maximum_gradient = max(maximum_gradient, gradient)
            require(gradient < 2e-6, "Calibrator not at recorded regularized optimum")
            negative_slopes += record["slope"] < 0
            counts += 1
        for stage in ["outer", 0, 1, 2]:
            records_here = [
                r
                for r in records
                if (
                    r["stage"] == "outer"
                    if stage == "outer"
                    else r["stage"] == "inner" and r["inner_fold"] == stage
                )
            ]
            targets = pd.DataFrame(records_here[0]["validation_cases"])
            targets["snapshot"] = pd.to_datetime(targets.snapshot)
            saved = (
                outer[outer.strategy.eq(CANDIDATE)]
                if stage == "outer"
                else inner[inner.candidate.eq(CANDIDATE) & inner.inner_fold.eq(stage)]
            )
            aligned = targets.merge(saved, on=KEYS, validate="one_to_one")
            np.testing.assert_array_equal(
                aligned.score,
                np.mean([r["mapped_scores"] for r in records_here], axis=0),
            )
        for arm in ["baseline", CANDIDATE]:
            inside, outside = (
                inner[inner.candidate.eq(arm)],
                outer[outer.strategy.eq(arm)],
            )
            require(
                outside.cutoff.eq(independent_cutoff(inside.churn, inside.score)).all(),
                "Cutoff differs",
            )
    metrics = pd.read_csv(child / "operating_metrics.csv")
    checked_metrics = 0
    for _, row in metrics.iterrows():
        rows = predictions[
            predictions.strategy.eq(row.strategy) & predictions.repeat.eq(row["repeat"])
        ]
        if row.scope == "outer_fold":
            rows = rows[rows.outer_fold.eq(row.outer_fold)]
        if row["slice"] in ["snapshot", "snapshot_activity"]:
            day = row["group"].split("/")[0]
            rows = rows[rows.snapshot.eq(day)]
        if row["slice"] in ["activity", "snapshot_activity"]:
            activity = row["group"].split("/")[-1]
            rows = rows[rows.n_orders_3m.gt(0).eq(activity == "recent_purchase")]
        require(
            confusion(rows) == [row[k] for k in ["tp", "fp", "fn", "tn"]],
            "Confusion counts differ",
        )
        require(
            len(rows) == row.rows and rows.StoreID.nunique() == row.resellers,
            "Population differs",
        )
        checked_metrics += 1
    # Reconstruct cluster bootstrap directly from case-level decision differences.
    paired = predictions[predictions.strategy.eq("baseline")].merge(
        predictions[predictions.strategy.eq(CANDIDATE)],
        on=KEYS + ["repeat", "outer_fold", "churn", "n_orders_3m"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    paired["tp_change"] = (
        paired.score_new.ge(paired.cutoff_new).astype(int)
        - paired.score_base.ge(paired.cutoff_base).astype(int)
    ) * paired.churn
    paired["fp_reduction"] = (
        paired.score_base.ge(paired.cutoff_base).astype(int)
        - paired.score_new.ge(paired.cutoff_new).astype(int)
    ) * (1 - paired.churn)
    stores = np.sort(paired.StoreID.unique())
    weights = np.random.default_rng(2026).multinomial(
        len(stores), np.repeat(1 / len(stores), len(stores)), size=5000
    )
    deltas, reductions = [], []
    for _, rows in paired.groupby("repeat"):
        values = (
            rows.groupby("StoreID")[["tp_change", "fp_reduction", "churn"]]
            .sum()
            .reindex(stores, fill_value=0)
        )
        sampled = weights @ values.to_numpy()
        deltas.append(sampled[:, 0] / sampled[:, 2])
        reductions.append(sampled[:, 1])
    bootstrap = pd.read_csv(child / "bootstrap.csv", float_precision="round_trip")
    np.testing.assert_array_equal(bootstrap.recall_difference, np.mean(deltas, axis=0))
    np.testing.assert_array_equal(bootstrap.fp_reduction, np.mean(reductions, axis=0))
    decision = json.loads((child / "decision.json").read_text())
    for name, values in [
        ("recall_difference_interval", bootstrap.recall_difference),
        ("fp_reduction_interval", bootstrap.fp_reduction),
    ]:
        np.testing.assert_array_equal(
            decision[name], np.quantile(values, [0.00025, 0.99975])
        )
    for filename, dimension in [
        ("period_safeguards.csv", ["snapshot", "activity"]),
        ("intersection_safeguards.csv", ["snapshot_activity"]),
    ]:
        safeguards = pd.read_csv(child / filename)
        if filename.startswith("intersection"):
            require(
                safeguards.passed.eq(
                    safeguards.recall_difference.ge(-0.05)
                    & safeguards.false_alert_reduction.ge(0)
                ).all(),
                "Intersection decision differs",
            )
    result = {
        "status": "passed",
        "component_identities": counts,
        "independent_operating_rows": checked_metrics,
        "cutoffs": 30,
        "maximum_logistic_gradient": maximum_gradient,
        "negative_slopes": negative_slopes,
        "bootstrap_draws": 5000,
        "classifiers_fitted": 0,
        "calibrators_fitted": 0,
        "verifier": input_identity({"script": Path(__file__)}),
    }
    destination = directory / "independent_verification.json"
    require(not destination.exists(), "Preserve previous verification")
    write_json(destination, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

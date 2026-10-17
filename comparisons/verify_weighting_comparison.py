"""Independently audit saved unit-weight comparisons without estimator refits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.grouped_precision_recall import BASE_PARAMETERS, FEATURES, MODEL_SEEDS
from comparisons.verify_basket_breadth import independent_cutoff
from comparisons.weighting_comparison import METHODS
from reseller_churn.artifacts import file_sha256, verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def ordered(frame):
    return frame.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def verify(directory):
    manifest = verify_manifest(directory)
    hgb = Path(manifest["inputs"]["cached_hgb_manifest"]["path"]).parent
    forest = Path(manifest["inputs"]["cached_forest_manifest"]["path"]).parent
    verify_manifest(hgb)
    verify_manifest(forest)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(hgb / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    predictions = read_csv(directory / "validation_predictions.csv")
    require(
        len(predictions) == 16500
        and not predictions.duplicated([*KEYS, "strategy", "repeat"]).any(),
        "Incomplete four-arm population",
    )
    cuts = 0
    fits = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        source_inner = read_csv(hgb / relative / "inner_scores.csv")
        source_outer = read_csv(hgb / relative / "predictions.csv")
        for current, source, identifier in [
            (inner, source_inner, "candidate"),
            (outer, source_outer, "strategy"),
        ]:
            pd.testing.assert_frame_equal(
                ordered(current[current[identifier].eq("baseline")]),
                ordered(source[source[identifier].eq("baseline")]),
            )
        rf = read_csv(forest / relative / "predictions.csv")
        rf = rf[rf.strategy.eq("random_forest")].assign(
            strategy="balanced_forest_reference"
        )
        pd.testing.assert_frame_equal(
            ordered(outer[outer.strategy.eq("balanced_forest_reference")]), ordered(rf)
        )
        base_inner = ordered(inner[inner.candidate.eq("baseline")])
        for arm in ("baseline", *METHODS):
            inside = ordered(inner[inner.candidate.eq(arm)])
            outside = ordered(outer[outer.strategy.eq(arm)])
            pd.testing.assert_frame_equal(
                inside[KEYS + ["churn"]], ordered(train)[KEYS + ["churn"]]
            )
            pd.testing.assert_frame_equal(
                outside[KEYS + ["churn", "n_orders_3m"]],
                ordered(valid)[KEYS + ["churn", "n_orders_3m"]],
            )
            require(
                inside.inner_fold.eq(base_inner.inner_fold).all()
                and inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner memberships differ",
            )
            require(
                set(inside.StoreID).isdisjoint(outside.StoreID),
                "Inner/outer reseller overlap",
            )
            require(
                outside.cutoff.eq(
                    independent_cutoff(inside.churn.to_numpy(), inside.score.to_numpy())
                ).all(),
                "Cutoff differs from inner objective",
            )
            cuts += 1
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 40, "Unexpected new fit count")
        for method in METHODS:
            rows = [row for row in identities if row["candidate"] == method]
            for stage in ["outer", 0, 1, 2]:
                group = (
                    [row for row in rows if row["stage"] == "outer"]
                    if stage == "outer"
                    else [
                        row
                        for row in rows
                        if row["stage"] == "inner" and row["inner_fold"] == stage
                    ]
                )
                require(
                    {row["seed"] for row in group} == set(MODEL_SEEDS),
                    "Ensemble membership differs",
                )
            for identity in rows:
                fitted = train
                if identity["stage"] == "inner":
                    held = inner.loc[
                        inner.candidate.eq(method)
                        & inner.inner_fold.eq(identity["inner_fold"]),
                        "StoreID",
                    ]
                    fitted = train[~train.StoreID.isin(held)]
                content = pd.util.hash_pandas_object(
                    fitted[[*KEYS, "churn", *FEATURES]], index=False
                ).values.tobytes()
                require(
                    identity["training_sha256"] == hashlib.sha256(content).hexdigest(),
                    "Training hash differs",
                )
                require(
                    identity["training_weighting"] == "unit/no sample_weight"
                    and identity["parameters"]["class_weight"] is None,
                    "Unexpected weighting metadata",
                )
                if method == "unweighted_hgb":
                    require(
                        all(
                            identity["parameters"][key] == value
                            for key, value in BASE_PARAMETERS.items()
                        ),
                        "HGB parameters changed",
                    )
                else:
                    params = identity["parameters"]
                    require(
                        params["n_estimators"] == 300
                        and params["max_depth"] is None
                        and params["min_samples_leaf"] == 1
                        and params["max_features"] == "sqrt"
                        and params["n_jobs"] == 1,
                        "RF parameters changed",
                    )
                    np.testing.assert_allclose(
                        identity["imputer_statistics"],
                        fitted[FEATURES].median().to_numpy(),
                        rtol=0,
                        atol=0,
                    )
                    require(
                        identity["missing_indicator_columns"]
                        == np.flatnonzero(fitted[FEATURES].isna().any()).tolist(),
                        "Imputer indicators differ",
                    )
                fits += 1
    refs = json.loads((directory / "assessment_manifests.json").read_text())
    for method in METHODS:
        assessment = directory / method
        require(
            file_sha256(assessment / "manifest.json") == refs[method]["sha256"],
            "Assessment manifest differs",
        )
        verify_manifest(assessment)
        metrics = pd.read_csv(assessment / "metrics.csv")
        checked = 0
        for scope, columns in [
            ("repeat_pooled", ["repeat", "strategy"]),
            ("outer_fold", ["repeat", "strategy", "outer_fold"]),
        ]:
            for identity, part in predictions[
                predictions.strategy.isin(["baseline", method])
            ].groupby(columns):
                selector = metrics.scope.eq(scope) & metrics.slice.eq("overall")
                for key, value in zip(columns, identity):
                    selector &= metrics["policy" if key == "strategy" else key].eq(
                        value
                    )
                row = metrics[selector].squeeze()
                a, y = part.score.ge(part.cutoff), part.churn.eq(1)
                require(
                    [int(mask.sum()) for mask in [a & y, a & ~y, ~a & y, ~a & ~y]]
                    == [row.tp, row.fp, row.fn, row.tn],
                    "Confusion counts differ",
                )
                checked += 1
        result = {
            "status": "passed",
            "outer_splits": 15,
            "inner_cutoffs_independently_reproduced": cuts,
            "recorded_training_identities_checked": fits,
            "overall_metric_rows": checked,
            "models_fitted_by_verifier": 0,
            "october_outcomes_accessed": False,
        }
        write_json(assessment / "independent_verification.json", result)
    write_json(
        directory / "independent_verification.json",
        {
            "status": "passed",
            "outer_splits": 15,
            "cutoffs": cuts,
            "new_training_identities": fits,
            "models_fitted": 0,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

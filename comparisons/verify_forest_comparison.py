"""No-fit audit of forest preprocessing, provenance, blend and operating metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.forest_comparison import BLEND, FOREST
from comparisons.grouped_precision_recall import FEATURES, MODEL_SEEDS
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import file_sha256, verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def ordered(rows):
    return rows.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def verify(directory):
    manifest = verify_manifest(directory)
    source = Path(manifest["inputs"]["cached_study_manifest"]["path"]).parent
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    require(protocol["model_seeds"] == list(MODEL_SEEDS), "Model seeds changed")
    predictions = read_csv(directory / "validation_predictions.csv")
    require(
        len(predictions) == 12375
        and not predictions.duplicated([*KEYS, "strategy", "repeat"]).any(),
        "Incomplete three-arm population",
    )
    checked_cutoffs = 0
    checked_identities = 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        repeat, fold = split["repeat"], split["outer_fold"]
        relative = Path(f"repeat_{repeat}/fold_{fold}")
        folder = directory / relative
        verify_manifest(folder)
        verify_manifest(source / relative)
        inner = read_csv(folder / "inner_scores.csv")
        outer = read_csv(folder / "predictions.csv")
        source_inner = read_csv(source / relative / "inner_scores.csv")
        source_outer = read_csv(source / relative / "predictions.csv")
        for current, reference, identifier in [
            (inner, source_inner, "candidate"),
            (outer, source_outer, "strategy"),
        ]:
            b = ordered(current[current[identifier].eq("baseline")])
            old = ordered(reference[reference[identifier].eq("baseline")])
            pd.testing.assert_frame_equal(b, old)
            forest = ordered(current[current[identifier].eq(FOREST)])
            blend = ordered(current[current[identifier].eq(BLEND)])
            pd.testing.assert_frame_equal(b[KEYS + ["churn"]], forest[KEYS + ["churn"]])
            pd.testing.assert_frame_equal(b[KEYS + ["churn"]], blend[KEYS + ["churn"]])
            np.testing.assert_allclose(
                blend.score, (b.score + forest.score) / 2, rtol=0, atol=0
            )
        for arm in ("baseline", FOREST, BLEND):
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
                inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner reseller overlap",
            )
            require(
                set(inside.StoreID).isdisjoint(outside.StoreID),
                "Outer reseller in training selection",
            )
            cutoff = independent_cutoff(
                inside.churn.to_numpy(), inside.score.to_numpy()
            )
            require(outside.cutoff.eq(cutoff).all(), "Inner cutoff mismatch")
            checked_cutoffs += 1
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 20, "Unexpected new fit count")
        for stage, group in pd.DataFrame(identities).groupby("stage"):
            if stage == "outer":
                require(set(group.seed) == set(MODEL_SEEDS), "Outer ensemble differs")
            else:
                for _, subgroup in group.groupby("inner_fold"):
                    require(
                        set(subgroup.seed) == set(MODEL_SEEDS), "Inner ensemble differs"
                    )
        assignment = inner[inner.candidate.eq(FOREST)]
        for identity in identities:
            fitted = train
            if identity["stage"] == "inner":
                held = assignment.loc[
                    assignment.inner_fold.eq(identity["inner_fold"]), "StoreID"
                ]
                fitted = train[~train.StoreID.isin(held)]
            content = pd.util.hash_pandas_object(
                fitted[[*KEYS, "churn", *FEATURES]], index=False
            ).values.tobytes()
            require(
                hashlib.sha256(content).hexdigest() == identity["training_sha256"],
                "Training hash differs",
            )
            expected = fitted[FEATURES].median().to_numpy()
            observed = np.array(
                [np.nan if x is None else x for x in identity["imputer_statistics"]]
            )
            np.testing.assert_allclose(
                observed, expected, rtol=0, atol=0, equal_nan=True
            )
            require(
                identity["missing_indicator_columns"]
                == np.flatnonzero(fitted[FEATURES].isna().any()).tolist(),
                "Imputer indicators differ",
            )
            require(
                identity["training_rows"] == len(fitted)
                and identity["training_positives"] == int(fitted.churn.sum()),
                "Training counts differ",
            )
            params = identity["forest_parameters"]
            require(
                params["n_estimators"] == 300
                and params["max_depth"] is None
                and params["min_samples_leaf"] == 1
                and params["max_features"] == "sqrt"
                and params["class_weight"] is None
                and params["bootstrap"]
                and params["n_jobs"] == 1,
                "Forest configuration differs",
            )
            checked_identities += 1
    assessment_refs = json.loads((directory / "assessment_manifests.json").read_text())
    for arm in (FOREST, BLEND):
        assessment = directory / arm
        require(
            file_sha256(assessment / "manifest.json") == assessment_refs[arm]["sha256"],
            "Assessment manifest differs",
        )
        verify_manifest(assessment)
        metric_table = pd.read_csv(assessment / "metrics.csv")
        checked_metrics = 0
        for scope, groups in [
            ("repeat_pooled", ["repeat", "strategy"]),
            ("outer_fold", ["repeat", "strategy", "outer_fold"]),
        ]:
            for identity, part in predictions[
                predictions.strategy.isin(["baseline", arm])
            ].groupby(groups):
                selector = metric_table.scope.eq(scope) & metric_table.slice.eq(
                    "overall"
                )
                for col, value in zip(groups, identity):
                    selector &= metric_table["policy" if col == "strategy" else col].eq(
                        value
                    )
                row = metric_table[selector].squeeze()
                alert = part.score.ge(part.cutoff)
                positive = part.churn.eq(1)
                counts = [
                    int(x.sum())
                    for x in [
                        alert & positive,
                        alert & ~positive,
                        ~alert & positive,
                        ~alert & ~positive,
                    ]
                ]
                require(
                    counts == [row.tp, row.fp, row.fn, row.tn],
                    "Confusion counts differ",
                )
                checked_metrics += 1
        result = {
            "status": "passed",
            "outer_splits": 15,
            "inner_cutoffs_independently_reproduced": checked_cutoffs,
            "recorded_forest_training_and_imputation_checks": checked_identities,
            "overall_metric_rows": checked_metrics,
            "models_fitted_by_verifier": 0,
            "october_outcomes_accessed": False,
        }
        write_json(assessment / "independent_verification.json", result)
    write_json(
        directory / "independent_verification.json",
        {
            "status": "passed",
            "inner_cutoffs": checked_cutoffs,
            "forest_training_identities": checked_identities,
            "two_assessments_checked": True,
            "models_fitted": 0,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

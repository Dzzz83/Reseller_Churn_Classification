"""Audit training-reference isolation and percentile construction without fits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.crossfit_percentile import FEATURES, NORMALIZED, RAW
from comparisons.grouped_precision_recall import MODEL_SEEDS
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]


def ordered(frame):
    return frame.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def independent_percentiles(reference, scores):
    knots = np.sort(np.unique(reference))
    midpoint = {
        point: (np.sum(reference < point) + 0.5 * np.sum(reference == point))
        / len(reference)
        for point in knots
    }
    result = []
    for score in scores:
        if score < knots[0]:
            value = 0.0
        elif score > knots[-1]:
            value = 1.0
        elif score in midpoint:
            value = midpoint[score]
        else:
            right = int(np.searchsorted(knots, score))
            low, high = knots[right - 1], knots[right]
            value = midpoint[low] + (score - low) / (high - low) * (
                midpoint[high] - midpoint[low]
            )
        result.append(value)
    return np.array(result)


def verify(directory):
    manifest = verify_manifest(directory)
    source = Path(manifest["inputs"]["source"]["path"]).parent
    verify_manifest(source)
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    predictions = read_csv(directory / "validation_predictions.csv")
    require(
        len(predictions) == 12375
        and not predictions.duplicated(KEYS + ["strategy", "repeat"]).any(),
        "Incomplete three-arm population",
    )
    fits, cuts = 0, 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner, outer = (
            read_csv(folder / "inner_scores.csv"),
            read_csv(folder / "predictions.csv"),
        )
        for current, original, column in [
            (inner, read_csv(source / relative / "inner_scores.csv"), "candidate"),
            (outer, read_csv(source / relative / "predictions.csv"), "strategy"),
        ]:
            pd.testing.assert_frame_equal(
                ordered(current[current[column].eq("baseline")]),
                ordered(original[original[column].eq("baseline")]),
            )
        base_inner = inner[inner.candidate.eq("baseline")]
        for arm in ["baseline", RAW, NORMALIZED]:
            inside, outside = (
                ordered(inner[inner.candidate.eq(arm)]),
                ordered(outer[outer.strategy.eq(arm)]),
            )
            pd.testing.assert_frame_equal(
                inside[KEYS + ["churn"]], ordered(train)[KEYS + ["churn"]]
            )
            pd.testing.assert_frame_equal(
                outside[KEYS + ["churn"]], ordered(valid)[KEYS + ["churn"]]
            )
            require(
                inside.inner_fold.eq(ordered(base_inner).inner_fold).all()
                and inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner membership differs",
            )
            require(
                outside.cutoff.eq(
                    independent_cutoff(inside.churn.to_numpy(), inside.score.to_numpy())
                ).all(),
                "Cutoff differs",
            )
            cuts += 1
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 40, "Unexpected component fit count")
        for stage in ["outer", 0, 1, 2]:
            if stage == "outer":
                base, held = train, valid
                records = [row for row in identities if row["stage"] == "outer"]
                saved = outer
                column = "strategy"
            else:
                inside = base_inner[base_inner.inner_fold.eq(stage)]
                held_stores = set(inside.StoreID)
                base, held = (
                    train[~train.StoreID.isin(held_stores)],
                    train[train.StoreID.isin(held_stores)],
                )
                records = [
                    row
                    for row in identities
                    if row["stage"] == "inner" and row["inner_fold"] == stage
                ]
                saved = inner[inner.inner_fold.eq(stage)]
                column = "candidate"
            halves = list(
                StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
                    base, base.churn, base.StoreID
                )
            )
            raw, mapped = [], []
            for half in [0, 1]:
                subset = [row for row in records if row["half"] == half]
                require(
                    {row["seed"] for row in subset} == set(MODEL_SEEDS)
                    and len(subset) == 5,
                    "Component seed membership differs",
                )
                ti, ri = halves[half]
                fitting, reference = base.iloc[ti], base.iloc[ri]
                require(
                    set(fitting.StoreID).isdisjoint(reference.StoreID)
                    and set(base.StoreID).isdisjoint(held.StoreID),
                    "Reseller reference leakage",
                )
                for identity in subset:
                    refs = pd.DataFrame(identity["reference_cases"])
                    refs["snapshot"] = pd.to_datetime(refs.snapshot)
                    targets = pd.DataFrame(identity["validation_cases"])
                    targets["snapshot"] = pd.to_datetime(targets.snapshot)
                    pd.testing.assert_frame_equal(
                        refs, reference[KEYS].reset_index(drop=True)
                    )
                    pd.testing.assert_frame_equal(
                        targets, held[KEYS].reset_index(drop=True)
                    )
                    content = pd.util.hash_pandas_object(
                        fitting[[*KEYS, "churn", *FEATURES]], index=False
                    ).values.tobytes()
                    require(
                        hashlib.sha256(content).hexdigest()
                        == identity["training_sha256"],
                        "Component training hash differs",
                    )
                    ref_score = np.array(identity["reference_scores"])
                    raw_score = np.array(identity["validation_raw_scores"])
                    require(
                        len(ref_score) == len(reference)
                        and np.isfinite(ref_score).all()
                        and np.all((ref_score >= 0) & (ref_score <= 1)),
                        "Invalid reference scores",
                    )
                    expected = independent_percentiles(ref_score, raw_score)
                    np.testing.assert_allclose(
                        identity["validation_percentiles"],
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    raw.append(raw_score)
                    mapped.append(np.array(identity["validation_percentiles"]))
                    fits += 1
            for arm, values in [(RAW, raw), (NORMALIZED, mapped)]:
                rows = (
                    saved[saved[column].eq(arm)]
                    .set_index(KEYS)
                    .reindex(pd.MultiIndex.from_frame(held[KEYS]))
                )
                np.testing.assert_allclose(
                    rows.score, np.mean(values, axis=0), rtol=0, atol=0
                )
    refs = json.loads((directory / "assessment_manifests.json").read_text())
    for arm in [RAW, NORMALIZED]:
        child = directory / arm
        require(
            file_sha256(child / "manifest.json") == refs[arm]["sha256"],
            "Assessment identity differs",
        )
        verify_manifest(child)
        metrics = pd.read_csv(child / "metrics.csv")
        count = 0
        for scope, columns in [
            ("repeat_pooled", ["repeat", "strategy"]),
            ("outer_fold", ["repeat", "strategy", "outer_fold"]),
        ]:
            for values, part in predictions[
                predictions.strategy.isin(["baseline", arm])
            ].groupby(columns):
                mask = metrics.scope.eq(scope) & metrics.slice.eq("overall")
                for key, value in zip(columns, values):
                    mask &= metrics["policy" if key == "strategy" else key].eq(value)
                row = metrics[mask].squeeze()
                a, y = part.score.ge(part.cutoff), part.churn.eq(1)
                require(
                    [int(m.sum()) for m in [a & y, a & ~y, ~a & y, ~a & ~y]]
                    == [row.tp, row.fp, row.fn, row.tn],
                    "Operating counts differ",
                )
                count += 1
        result = dict(
            status="passed",
            component_identities=fits,
            cutoffs=cuts,
            overall_metric_rows=count,
            models_fitted=0,
            october_outcomes_accessed=False,
            verifier=input_identity(dict(script=Path(__file__))),
        )
        write_json(child / "independent_verification.json", result)
    write_json(
        directory / "independent_verification.json",
        dict(status="passed", component_identities=fits, cutoffs=cuts, models_fitted=0),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

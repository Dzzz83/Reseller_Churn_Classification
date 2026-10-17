"""Audit saved breadth-study predictions, splits and decisions without refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from reseller_churn.artifacts import verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def independent_cutoff(y, scores):
    """Sorted cumulative counts with intact ties; no call to study selector."""
    grouped = (
        pd.DataFrame({"y": y, "score": scores})
        .groupby("score")
        .y.agg(["sum", "size"])
        .sort_index(ascending=False)
    )
    grouped["tp"] = grouped["sum"].cumsum()
    grouped["alerts"] = grouped["size"].cumsum()
    choices = [
        (row.tp / row.alerts, row.tp / np.sum(y), float(cutoff))
        for cutoff, row in grouped.iterrows()
        if row.tp / np.sum(y) > 0.9
    ]
    choices.append((np.mean(y), 1.0, 0.0))
    return max(choices)[2]


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    data = read_csv(directory / "development_input.csv")
    predictions = read_csv(directory / "validation_predictions.csv")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    require(len(predictions) == 8250, "Incomplete paired predictions")
    require(
        not predictions.duplicated([*KEYS, "repeat", "strategy"]).any(),
        "Duplicate validation prediction",
    )
    checked_cutoffs = 0
    for split in protocol["splits"]:
        repeat, fold = split["repeat"], split["outer_fold"]
        train, valid = validate_partition(data, split)
        folder = directory / f"repeat_{repeat}/fold_{fold}"
        verify_manifest(folder)
        inner = read_csv(folder / "inner_scores.csv")
        choices = pd.read_csv(
            folder / "inner_selections.csv", float_precision="round_trip"
        )
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 40, "Unexpected estimator count")
        for arm in [item["id"] for item in protocol["arms"]]:
            outer = predictions[
                (predictions.repeat == repeat)
                & (predictions.outer_fold == fold)
                & predictions.strategy.eq(arm)
            ]
            outer = outer.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(
                outer[KEYS + ["churn", "n_orders_3m"]],
                valid[KEYS + ["churn", "n_orders_3m"]].reset_index(drop=True),
            )
            inside = inner[inner.candidate.eq(arm)].sort_values(["snapshot", "StoreID"])
            pd.testing.assert_frame_equal(
                inside[KEYS + ["churn"]].reset_index(drop=True),
                train[KEYS + ["churn"]].reset_index(drop=True),
            )
            require(
                inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner reseller leakage",
            )
            require(
                set(inside.StoreID).isdisjoint(valid.StoreID),
                "Outer labels in inner data",
            )
            specification = next(item for item in protocol["arms"] if item["id"] == arm)
            for identity in [row for row in identities if row["candidate"] == arm]:
                fitted_rows = train
                if identity["stage"] == "inner":
                    held_stores = inside.loc[
                        inside.inner_fold.eq(identity["inner_fold"]), "StoreID"
                    ]
                    fitted_rows = train[~train.StoreID.isin(held_stores)]
                content = pd.util.hash_pandas_object(
                    fitted_rows[[*KEYS, "churn", *specification["features"]]],
                    index=False,
                ).values.tobytes()
                require(
                    identity["training_sha256"] == hashlib.sha256(content).hexdigest(),
                    "Fitting content differs from isolated training rows",
                )
            cutoff = independent_cutoff(
                inside.churn.to_numpy(), inside.score.to_numpy()
            )
            require(outer.cutoff.eq(cutoff).all(), "Invalid training-selected cutoff")
            require(
                float(choices.loc[choices.candidate.eq(arm), "cutoff"].iloc[0])
                == cutoff,
                "Selection table differs",
            )
            checked_cutoffs += 1
    checked_metrics = 0
    for scope, columns in [
        ("repeat_pooled", ["repeat", "strategy"]),
        ("outer_fold", ["repeat", "strategy", "outer_fold"]),
    ]:
        for identity, group in predictions.groupby(columns):
            selector = metrics.scope.eq(scope) & metrics.slice.eq("overall")
            for column, value in zip(columns, identity):
                selector &= metrics["policy" if column == "strategy" else column].eq(
                    value
                )
            record = metrics[selector].squeeze()
            positive, alert = group.churn.eq(1), group.score.ge(group.cutoff)
            counts = [
                int(mask.sum())
                for mask in [
                    positive & alert,
                    ~positive & alert,
                    positive & ~alert,
                    ~positive & ~alert,
                ]
            ]
            require(
                counts == [int(record[name]) for name in ["tp", "fp", "fn", "tn"]],
                "Saved confusion counts differ",
            )
            checked_metrics += 1
    result = {
        "status": "passed",
        "outer_splits": 15,
        "inner_cutoffs_independently_reproduced": checked_cutoffs,
        "overall_metric_rows_independently_reproduced": checked_metrics,
        "validation_predictions": len(predictions),
        "models_fitted": 0,
        "october_outcomes_accessed": False,
    }
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)


if __name__ == "__main__":
    main()

"""No-fit reconstruction of candidate identities, thresholds and October confusion counts."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from comparisons.all_candidate_precision import NAMES, STUDIES, ordered, score_models
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.development_checks import validate_development_cases
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]


def enumerate_choice(labels, scores, target):
    y = np.asarray(labels) == 1
    values = np.asarray(scores)
    eligible = []
    for cut in sorted(set([0.0, *values])):
        alert = values >= cut
        tp = int((y & alert).sum())
        fp = int((~y & alert).sum())
        if tp + fp and tp * 100 >= target * (tp + fp):
            eligible.append((tp, Fraction(tp, tp + fp), cut))
    return None if not eligible else max(eligible)[2]


def verify(directory, outcomes=False):
    verify_manifest(directory)
    pref = json.loads((directory / "preexecution_identity.json").read_text())
    for identity in pref["code"].values():
        require(
            file_sha256(directory / "source_snapshot" / Path(identity["path"]).name)
            == identity["sha256"],
            "Frozen code changed",
        )
    data = read_csv(directory / "development_input.csv")
    validate_development_cases(data)
    require(
        (len(data), int(data.churn.sum()), data.StoreID.nunique()) == (1375, 212, 422),
        "Wrong data",
    )
    oof = read_csv(directory / "development_oof.csv")
    test = read_csv(directory / "unlabeled_features.csv")
    predictions = read_csv(directory / "unlabeled_predictions.csv")
    selections = pd.read_csv(
        directory / "development_selections.csv", float_precision="round_trip"
    )
    require(
        set(oof.candidate) == set(NAMES) and len(selections) == 117,
        "Missing candidate or target",
    )
    for name, rows in oof.groupby("candidate"):
        pd.testing.assert_frame_equal(
            ordered(rows)[KEYS + ["churn"]], ordered(data)[KEYS + ["churn"]]
        )
        for _, selection in selections[selections.candidate.eq(name)].iterrows():
            cut = enumerate_choice(
                rows.churn, rows.score, int(selection.minimum_precision_percent)
            )
            require(
                (cut is None and not selection.feasible)
                or (cut is not None and selection.feasible and cut == selection.cutoff),
                "Independent target cutoff differs",
            )
            if cut is not None:
                chosen = rows.score.ge(cut)
                tp = int((chosen & rows.churn.eq(1)).sum())
                fp = int((chosen & rows.churn.eq(0)).sum())
                require(
                    tp == selection.tp and fp == selection.fp,
                    "Development counts differ",
                )
                part = predictions[
                    predictions.candidate.eq(name)
                    & predictions.minimum_precision_percent.eq(
                        selection.minimum_precision_percent
                    )
                ]
                pd.testing.assert_frame_equal(ordered(part)[KEYS], ordered(test)[KEYS])
                require(
                    part.cutoff.eq(cut).all()
                    and part.predicted_churn.eq(part.score.ge(cut)).all(),
                    "Frozen test cutoff differs",
                )
    splits = json.loads((directory / "inner_splits.json").read_text())["folds"]
    fit_records = 0
    for name in STUDIES:
        folder = directory / name
        verify_manifest(folder)
        bundle = joblib.load(folder / "frozen_models.joblib")
        columns = bundle["spec"]["features"]
        expected = score_models(bundle, test)
        saved = pd.read_csv(
            folder / "full_scores.csv", float_precision="round_trip"
        ).score
        np.testing.assert_allclose(expected, saved, rtol=0, atol=1e-15)
        for _, part in predictions[predictions.candidate.eq(name)].groupby(
            "minimum_precision_percent"
        ):
            np.testing.assert_allclose(
                ordered(part).score, expected, rtol=0, atol=1e-15
            )
        records = json.loads((folder / "training_identities.json").read_text())["fits"]
        for record in records:
            fitting = (
                data
                if record["stage"] == "full"
                else data.iloc[splits[record["inner_fold"]]["train"]]
            )
            if record["stage"] == "inner":
                held = data.iloc[splits[record["inner_fold"]]["validation"]]
                require(
                    set(fitting.StoreID).isdisjoint(held.StoreID), "Inner reseller leak"
                )
            content = pd.util.hash_pandas_object(
                fitting[KEYS + ["churn"] + columns], index=False
            ).values.tobytes()
            require(
                hashlib.sha256(content).hexdigest() == record["training_sha256"],
                "Fitting cases/features differ",
            )
            if name in [
                "reseller_weighted",
                "period_class_weighted",
                "activity_experts",
            ]:
                if name == "reseller_weighted":
                    inverse = (
                        1
                        / fitting.groupby("StoreID")
                        .StoreID.transform("size")
                        .to_numpy()
                    )
                    totals = [
                        inverse[fitting.churn.to_numpy() == label].sum()
                        for label in [0, 1]
                    ]
                    weights = (
                        inverse
                        * len(fitting)
                        / (2 * np.array(totals)[fitting.churn.to_numpy()])
                    )
                elif name == "period_class_weighted":
                    counts = fitting.groupby(["snapshot", "churn"]).size()
                    weights = np.array(
                        [
                            len(fitting) / (8 * counts.loc[(r.snapshot, r.churn)])
                            for r in fitting.itertuples()
                        ]
                    )
                else:
                    counts = fitting.churn.value_counts()
                    weights = np.array(
                        [len(fitting) / (2 * counts.loc[y]) for y in fitting.churn]
                    )
                    weights = weights[
                        fitting.n_orders_3m.gt(0).eq(record["recent_purchase"])
                    ]
                np.testing.assert_allclose(
                    record["weights"], weights, rtol=1e-14, atol=1e-15
                )
            fit_records += 1
    rows_verified = 0
    if outcomes:
        verify_manifest(directory / "evaluation")
        cases = read_csv(directory / "evaluation/evaluated_cases.csv")
        metrics = pd.read_csv(
            directory / "evaluation/metrics.csv", float_precision="round_trip"
        )
        labels_path = Path(
            json.loads((directory / "evaluation/manifest.json").read_text())["inputs"][
                "supplied_outcomes"
            ]["path"]
        )
        labels = read_csv(labels_path)[KEYS + ["churn"]]
        for strategy, part in cases.groupby("strategy"):
            expected = predictions[predictions.strategy.eq(strategy)].merge(
                labels, on=KEYS, validate="one_to_one"
            )
            pd.testing.assert_frame_equal(
                ordered(part), ordered(expected)[part.columns]
            )
        for _, row in metrics.iterrows():
            part = cases[
                cases.candidate.eq(row.candidate)
                & cases.minimum_precision_percent.eq(row.minimum_precision_percent)
            ]
            if row.subgroup != "overall":
                part = part[
                    part.n_orders_3m.gt(0).eq(row.subgroup == "recent_purchase")
                ]
            y = part.churn.eq(1)
            a = part.score.ge(part.cutoff)
            tp = int((y & a).sum())
            fp = int((~y & a).sum())
            fn = int((y & ~a).sum())
            tn = int((~y & ~a).sum())
            for key, value in dict(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                rows=len(part),
                churners=int(y.sum()),
                alerts=tp + fp,
            ).items():
                require(row[key] == value, "Independent confusion differs: " + key)
            require(
                row.precision_defined == bool(tp + fp > 0), "Precision flag differs"
            )
            require(
                (tp + fp == 0 and pd.isna(row.precision))
                or row.precision == tp / (tp + fp),
                "Precision differs",
            )
            require(row.recall == tp / (tp + fn), "Recall differs")
            rows_verified += 1
    destination = directory / (
        "outcome_audit.json" if outcomes else "preparation_audit.json"
    )
    require(not destination.exists(), "Preserve audit")
    result = dict(
        status="passed",
        candidate_procedures=len(NAMES),
        precision_targets=len(selections),
        training_records=fit_records,
        full_model_bundles=len(STUDIES),
        test_metric_rows=rows_verified,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--outcomes", action="store_true")
    args = parser.parse_args()
    verify(args.directory, args.outcomes)
